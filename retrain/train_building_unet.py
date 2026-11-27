"""Train a ResNet18-UNet building localization model on the 39k building-extraction pairs.

This is the first stage of the xView2-winner-style two-stage pipeline: a segmentation
model that localizes buildings on the pre-disaster image. The checkpoint is later used
to score/filter TTA proposals and to derive building masks for grading.

Usage:
  python train_building_unet.py --steps 2500 --batch 12 --crop 256 --out run/building_unet
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from inventory import ROOT

DATA = Path(r"F:/transformer/building train")
PRETRAINED = Path(r"D:/tools/weights/resnet18-f37072fd.pth")


class Pairs(Dataset):
    def __init__(self, images, labels, crop, train, repeats=1):
        self.images, self.labels = images, labels
        self.crop, self.train = crop, train
        self.repeats = repeats

    def __len__(self):
        return len(self.images) * self.repeats

    def __getitem__(self, index):
        index %= len(self.images)
        image = cv2.imread(str(self.images[index]), cv2.IMREAD_COLOR)
        label = cv2.imread(str(self.labels[index]), cv2.IMREAD_UNCHANGED)
        if image is None or label is None:
            raise ValueError(f"bad pair {index}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if label.ndim == 3:
            label = label[:, :, 0]
        if label.shape[:2] != image.shape[:2]:
            label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (label > 127).astype(np.float32)
        height, width = image.shape[:2]
        if self.train:
            top = np.random.randint(0, max(1, height - self.crop + 1))
            left = np.random.randint(0, max(1, width - self.crop + 1))
            image = image[top:top + self.crop, left:left + self.crop]
            mask = mask[top:top + self.crop, left:left + self.crop]
            if np.random.rand() < .5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if np.random.rand() < .5:
                image, mask = image[::-1], mask[::-1]
            turns = np.random.randint(0, 4)
            if turns:
                image = np.rot90(image, turns)
                mask = np.rot90(mask, turns)
        else:
            top = (height - self.crop) // 2
            left = (width - self.crop) // 2
            image = image[top:top + self.crop, left:left + self.crop]
            mask = mask[top:top + self.crop, left:left + self.crop]
        image = np.ascontiguousarray(image, np.float32) / 255.
        image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(np.ascontiguousarray(mask))


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision
        backbone = torchvision.models.resnet18(weights=None)
        state = torch.load(PRETRAINED, map_location="cpu")
        backbone.load_state_dict(state, strict=False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)   # 128
        self.pool = backbone.maxpool                                          # 64
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2            # 64, 32
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4            # 16, 8
        self.up4 = conv_block(512 + 256, 256)
        self.up3 = conv_block(256 + 128, 128)
        self.up2 = conv_block(128 + 64, 64)
        self.up1 = conv_block(64, 32)
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s = self.stem(x)
        p = self.pool(s)
        c1 = self.layer1(p)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        d = F.interpolate(c4, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.up4(torch.cat((d, c3), dim=1))
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.up3(torch.cat((d, c2), dim=1))
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.up2(torch.cat((d, c1), dim=1))
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.up1(d)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        return self.final(d)


def dice_score(logits, target):
    probability = logits.sigmoid()
    intersection = (probability * target).sum()
    return (2 * intersection + 1.) / (probability.sum() + target.sum() + 1.)


def validate(model, loader, device, limit=40):
    model.eval()
    dice, iou = [], []
    with torch.no_grad():
        for index, (image, target) in enumerate(loader):
            if index >= limit:
                break
            image, target = image.to(device), target.to(device)
            logits = model(image)
            probability = logits[:, 0].sigmoid() > .5
            intersection = (probability & (target > .5)).sum().item()
            union = (probability | (target > .5)).sum().item()
            dice.append(2 * intersection / max(1, probability.sum().item() + (target > .5).sum().item()))
            iou.append(intersection / max(1, union))
    model.train()
    return float(np.mean(dice)), float(np.mean(iou))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--out", type=Path, default=ROOT / "run/building_unet")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(DATA.joinpath("image").glob("*.tif"))
    train_files = [path for index, path in enumerate(files) if index % 50 != 0]
    val_files = [path for index, path in enumerate(files) if index % 50 == 0]
    train_set = Pairs(train_files, [DATA / "label" / path.name for path in train_files], args.crop, True,
                      repeats=1)
    val_set = Pairs(val_files, [DATA / "label" / path.name for path in val_files], args.crop, False)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True,
                              pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    args.out.mkdir(parents=True, exist_ok=True)
    best = -1.
    start = time.monotonic()
    step = 0
    print(f"train {len(train_files)} | val {len(val_files)} | crop {args.crop} | batch {args.batch}", flush=True)
    for image, target in train_loader:
        if step >= args.steps or time.monotonic() - start > args.seconds:
            break
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(image)
            bce = F.binary_cross_entropy_with_logits(logits[:, 0], target)
            dice = 1 - dice_score(logits[:, 0], target)
            loss = bce + dice
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        if step % 50 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.4f} bce {bce.item():.4f} "
                  f"lr {scheduler.get_last_lr()[0]:.2e} elapsed {time.monotonic()-start:.0f}s", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            val_dice, val_iou = validate(model, val_loader, device)
            print(f"validation step {step}: dice {val_dice:.4f} iou {val_iou:.4f}", flush=True)
            if val_dice > best:
                best = val_dice
                torch.save({"model": model.state_dict(), "crop": args.crop, "step": step,
                            "val_dice": val_dice, "val_iou": val_iou,
                            "train_files": len(train_files)}, args.out / "building_unet.pt")
                print(f"saved best ({val_dice:.4f})", flush=True)
    state = torch.load(args.out / "building_unet.pt", map_location="cpu")
    metadata = {"crop": state["crop"], "step": state["step"], "val_dice": state["val_dice"],
                "val_iou": state["val_iou"], "train_files": state["train_files"],
                "elapsed_seconds": round(time.monotonic() - start, 1), "steps": args.steps}
    (args.out / "training.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
