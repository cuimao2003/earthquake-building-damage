"""Train building extraction, then fine-tune on the paired earthquake imagery."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import DamageTransformer
from prepare import DATA, SIZE, write_json


RUN = Path(__file__).resolve().parent / "run"


def save_checkpoint(path, model, **metadata):
    temp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), **metadata}, temp)
    temp.replace(path)


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def tensor_from_batch(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def image_tensor(batch, device):
    return tensor_from_batch(np.moveaxis(batch, -1, 1), device).float().div_(127.5).sub_(1.)


def augment_image(image, strength):
    scale = 1 + (torch.rand((len(image), 1, 1, 1), device=image.device) - 0.5) * strength
    bias = (torch.rand((len(image), 3, 1, 1), device=image.device) - 0.5) * strength
    return (image * scale + bias).clamp_(-1, 1)


def transform_chip(chip, rng):
    out = np.rot90(chip, int(rng.integers(4)), axes=(0, 1))
    return out[:, ::-1] if rng.random() < 0.5 else out


def footprint_loss(logits, target):
    logits = logits[:, 0].float()
    target = target.float()
    probability = logits.sigmoid()
    intersection = (probability * target).sum()
    dice = (2 * intersection + 1) / (probability.sum() + target.sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + 1 - dice


@torch.no_grad()
def building_validation(model, val, count, batch, device, amp):
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.int64)
    for start in range(0, count, batch):
        array = val[start:min(count, start + batch)]
        x = image_tensor(array[..., :3], device)
        y = tensor_from_batch(array[..., 3], device).bool()
        with torch.cuda.amp.autocast(enabled=amp):
            pred = model(x)[:, 0] > 0
        totals += torch.stack(((pred & y).sum(), pred.sum(), y.sum()))
    tp, pred, truth = totals.tolist()
    return {"f1": 2 * tp / max(1, pred + truth), "iou": tp / max(1, pred + truth - tp)}


def train_building(args, model, device, amp):
    manifest = json.loads((args.cache / "building.json").read_text(encoding="utf-8"))
    train_count, val_count = (len(manifest[k]["names"]) for k in ("train", "val"))
    train, val = (np.load(args.cache / f"building_{k}.npy", mmap_mode="r")
                  for k in ("train", "val"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0011, weight_decay=0.015)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    rng = np.random.default_rng(args.seed)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00012)
    history, best = [], -1.
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = torch.zeros((), device=device)
        order = rng.permutation(train_count)
        batches = 0
        for index in range(0, train_count, args.batch):
            ids = order[index:index + args.batch]
            array = np.stack([transform_chip(train[i], rng) for i in ids])
            x = augment_image(image_tensor(array[..., :3], device), 0.27)
            y = tensor_from_batch(array[..., 3], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = footprint_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach()
            batches += 1
            if time.time() >= args.deadline:
                break
        schedule.step()
        scores = building_validation(model, val, val_count, args.batch, device, amp)
        row = {"epoch": epoch, "batches": batches, "loss": float(total) / batches,
               **scores, "elapsed_seconds": round(time.monotonic() - start, 1)}
        if not np.isfinite(row["loss"]):
            raise RuntimeError("Non-finite building loss; retry using --no-amp")
        print(json.dumps(row), flush=True)
        history.append(row)
        if scores["f1"] > best:
            best = scores["f1"]
            save_checkpoint(args.run / "building_best.pt", model, stage="building", **row)
        write_json(args.run / "building_history.json", history)
        if time.time() >= args.deadline:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("building", "earthquake"))
    parser.add_argument("--cache", type=Path, default=DATA)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--steps", type=int, default=115, help="Earthquake updates per epoch")
    parser.add_argument("--seed", type=int, default=924)
    parser.add_argument("--seconds", type=float, default=5400)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--focus", action="store_true",
                        help="Use variable-scale building crops and hierarchical damage loss")
    parser.add_argument("--init", type=Path,
                        help="Weights for earthquake fine-tuning, defaults to the building checkpoint")
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.steps, args.seconds) <= 0:
        parser.error("epochs, batch, steps and seconds must be positive")
    if args.stage == "building" and (args.focus or args.init):
        parser.error("--focus and --init apply to earthquake fine-tuning")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    args.run.mkdir(parents=True, exist_ok=True)
    args.deadline = time.time() + args.seconds
    result = DamageTransformer().to(device)
    print(json.dumps({"stage": args.stage, "device": str(device), "amp": amp,
                      "parameters": sum(p.numel() for p in result.parameters())}), flush=True)
    if args.stage == "building":
        train_building(args, result, device, amp)
    else:
        from fit_damage import train_earthquake
        train_earthquake(args, result, device, amp)


if __name__ == "__main__":
    main()
