"""Fine-tune the building UNet on xBD earthquake pre-disaster imagery + building polygons.

The building-extraction pretraining data is aerial/Asia domain; a short fine-tune on
xBD (Mexico + Palu) building polygons adapts the mask to the target satellite imagery,
which is what the localization filter uses.

Usage:
  python finetune_building_unet_xbd.py --init run/building_unet/building_unet.pt \
      --steps 300 --batch 8 --crop 384 --out run/building_unet_xbd
"""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from shapely import wkt

import train_proposal_transformer as T
from inventory import ROOT
from train_building_unet import UNet, dice_score

CROP = 384


def read_pre_mask(label_path, shape=(1024, 1024)):
    labels = json.loads(Path(label_path).read_bytes())["features"]["xy"]
    sheet = np.zeros(shape, np.uint8)
    for item in labels:
        geometry = wkt.loads(item["wkt"])
        coords = np.asarray(geometry.exterior.coords, np.float32)
        cv2.fillPoly(sheet, [np.round(coords).astype(np.int32)], 1)
    return sheet


class XBDPairs(Dataset):
    def __init__(self, rows, crop, train, repeats=1):
        self.rows, self.crop, self.train, self.repeats = rows, crop, train, repeats

    def __len__(self):
        return len(self.rows) * self.repeats

    @lru_cache(maxsize=64)
    def mask_for(self, index):
        root = Path(self.rows[index]["source"])
        return read_pre_mask(root / "labels" / (self.rows[index]["chip"] + "_pre_disaster.json"))

    def __getitem__(self, index):
        index %= len(self.rows)
        row = self.rows[index]
        root = Path(row["source"])
        image = cv2.imread(str(root / "images" / (row["chip"] + "_pre_disaster.png")), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"bad image {row['chip']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.mask_for(index).astype(np.float32)
        height, width = image.shape[:2]
        if self.train:
            top = np.random.randint(0, max(1, height - self.crop + 1))
            left = np.random.randint(0, max(1, width - self.crop + 1))
        else:
            top = (height - self.crop) // 2
            left = (width - self.crop) // 2
        image = image[top:top + self.crop, left:left + self.crop]
        mask = mask[top:top + self.crop, left:left + self.crop]
        if self.train:
            if np.random.rand() < .5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if np.random.rand() < .5:
                image, mask = image[::-1], mask[::-1]
            turns = np.random.randint(0, 4)
            if turns:
                image = np.rot90(image, turns)
                mask = np.rot90(mask, turns)
        image = np.ascontiguousarray(image, np.float32) / 255.
        image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(np.ascontiguousarray(mask))


def validate(model, loader, device, limit=20):
    model.eval()
    dice = []
    with torch.no_grad():
        for index, (image, target) in enumerate(loader):
            if index >= limit:
                break
            logits = model(image.to(device))[:, 0]
            probability = logits.sigmoid() > .5
            truth = target.to(device) > .5
            intersection = (probability & truth).sum().item()
            dice.append(2 * intersection / max(1, probability.sum().item() + truth.sum().item()))
    model.train()
    return float(np.mean(dice))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, default=ROOT / "run/building_unet/building_unet.pt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest_plus_palu.json")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--crop", type=int, default=CROP)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "run/building_unet_xbd")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_bytes())
    rows = [row for row in manifest["train"] if Path(row["source"]).joinpath("labels").is_dir()]
    rng = np.random.default_rng(7)
    order = rng.permutation(len(rows))
    val_rows = [rows[i] for i in order[:24]]
    train_rows = [rows[i] for i in order[24:]]
    repeats = max(1, (args.steps * args.batch + len(train_rows) - 1) // len(train_rows) + 1)
    print(f"xBD building fine-tune: train {len(train_rows)} | val {len(val_rows)} | repeats {repeats}",
          flush=True)
    train_set = XBDPairs(train_rows, args.crop, True, repeats=repeats)
    val_set = XBDPairs(val_rows, args.crop, False)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True,
                              pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    args.out.mkdir(parents=True, exist_ok=True)
    best, start, step = -1., time.monotonic(), 0
    for image, target in train_loader:
        if step >= args.steps:
            break
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(image)
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], target) + 1 - dice_score(logits[:, 0], target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        if step % 100 == 0:
            val_dice = validate(model, val_loader, device)
            print(f"step {step}/{args.steps} loss {loss.item():.4f} val dice {val_dice:.4f} "
                  f"elapsed {time.monotonic()-start:.0f}s", flush=True)
            if val_dice > best:
                best = val_dice
                torch.save({"model": model.state_dict(), "crop": args.crop, "step": step,
                            "val_dice": val_dice, "init": str(args.init)}, args.out / "building_unet_xbd.pt")
    if best < 0:
        best = validate(model, val_loader, device)
        torch.save({"model": model.state_dict(), "crop": args.crop, "step": step,
                    "val_dice": best, "init": str(args.init)}, args.out / "building_unet_xbd.pt")
    metadata = {"init": str(args.init), "steps": step, "val_dice": best, "crop": args.crop,
                "elapsed_seconds": round(time.monotonic() - start, 1)}
    (args.out / "training.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
