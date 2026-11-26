"""Detect individual buildings on pre-earthquake imagery using center heatmaps."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely import wkt
import torch
from torch import nn
from torch.nn import functional as F

from features import read_rgb
from inventory import ROOT, write_json
from model import DamageTransformer

SIZE = 256
STRIDE = 2
SOURCE_CACHE = ROOT.parent / "cache/earthquake"
TARGET_CACHE = ROOT / "data/center_targets"


class CenterDetector(DamageTransformer):
    def __init__(self):
        super().__init__()
        del self.fusion, self.damage, self.damage_context
        self.center = nn.Conv2d(16, 1, 1)
        self.bounds = nn.Conv2d(16, 4, 1)
        nn.init.constant_(self.center.bias, -2.19)

    def forward(self, x):
        decoded = self.decode(self.encode(x))
        return self.center(decoded), self.bounds(decoded), self.footprint(decoded)


def building_boxes(record, post=False):
    date = "post" if post else "pre"
    path = Path(record["source"]) / "labels" / f"{record['chip']}_{date}_disaster.json"
    return np.asarray([wkt.loads(entry["wkt"]).bounds
                       for entry in json.loads(path.read_bytes())["features"]["xy"]],
                      dtype=np.float32).reshape(-1, 4)


def make_targets(boxes, shape):
    h, w = shape
    result = np.zeros((h // STRIDE, w // STRIDE, 6), np.float32)
    for l, t, r, b in boxes:
        bw, bh = max(1., r - l), max(1., b - t)
        x, y = np.clip([(l + r) / (2 * STRIDE), (t + b) / (2 * STRIDE)], 0,
                       [w // STRIDE - 1.001, h // STRIDE - 1.001])
        cx, cy = int(x), int(y)
        sigma = np.clip(min(bw, bh) / (STRIDE * 7), 1.1, 4.)
        radius = int(np.ceil(sigma * 3))
        x0, y0, x1, y1 = max(0, cx-radius), max(0, cy-radius), min(w//STRIDE, cx+radius+1), min(h//STRIDE, cy+radius+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        values = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        area = result[y0:y1, x0:x1, 0]
        np.maximum(area, values, out=area)
        result[cy, cx, 1:5] = [np.log(bw / STRIDE), np.log(bh / STRIDE), x - cx, y - cy]
        xa, ya, xb, yb = np.rint(np.array([l, t, r, b]) / STRIDE).astype(np.int32)
        result[max(0, ya):min(h//STRIDE, yb), max(0, xa):min(w//STRIDE, xb), 5] = 1
    return result


@lru_cache(maxsize=14)
def cached_tile(chip, source):
    path = SOURCE_CACHE / f"{chip}.npy"
    image = (np.load(path, mmap_mode="r")[..., :3] if path.exists()
             else read_rgb(Path(source) / "images" / f"{chip}_pre_disaster.png"))
    target = np.load(TARGET_CACHE / f"{chip}.npy", mmap_mode="r")
    return image, target


def batch_crops(records, rng, batch):
    images = np.empty((batch, SIZE, SIZE, 3), np.uint8)
    targets = np.empty((batch, SIZE // STRIDE, SIZE // STRIDE, 6), np.float32)
    for im, label in zip(images, targets):
        r = records[rng.integers(len(records))]
        data, truth = cached_tile(r["chip"], r["source"])
        x, y = [int(rng.integers(d - SIZE // STRIDE + 1)) for d in (truth.shape[1], truth.shape[0])]
        im[:] = data[y * STRIDE:y * STRIDE + SIZE, x * STRIDE:x * STRIDE + SIZE]
        label[:] = truth[y:y + SIZE // STRIDE, x:x + SIZE // STRIDE]
    return images, targets


def inputs(images, device):
    return (torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
            .to(device).float().div_(127.5).sub_(1.))


def loss_for(det, data):
    heat, bbox, footprint = det
    heat, bbox, footprint = heat.float(), bbox.float(), footprint.float()
    labels = data[:, :1]
    peaks = labels == 1
    p = heat.sigmoid()
    focal = -torch.where(peaks, (1 - p).square() * F.logsigmoid(heat),
                         p.square() * F.logsigmoid(-heat) * (1 - labels).pow(4)).sum()
    n = peaks.sum().clamp_min(1.)
    regression = (F.smooth_l1_loss(bbox, data[:, 1:5], reduction="none") * peaks).sum() / n
    localization = F.binary_cross_entropy_with_logits(footprint, data[:, 5:6])
    return focal / n + regression + .18 * localization


def box_iou(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    overlap = np.prod(np.clip(hi - lo, 0, None), axis=-1)
    union = ((a[:, 2:] - a[:, :2]).prod(1)[:, None] +
             (b[:, 2:] - b[:, :2]).prod(1)[None, :] - overlap)
    return overlap / np.maximum(union, 1e-6)


def nms(boxes, scores, threshold=.38):
    order = scores.argsort()[::-1]
    kept = []
    while order.size:
        selected, order = order[0], order[1:]
        kept.append(selected)
        order = order[box_iou(boxes[selected:selected + 1], boxes[order])[0] <= threshold]
    return np.asarray(kept, dtype=int)


@torch.no_grad()
def detect(model, image, device, threshold=.22, batch=12):
    height, width = image.shape[:2]
    pad = ((0, max(0, SIZE - height)), (0, max(0, SIZE - width)), (0, 0))
    image = np.pad(image, pad, mode="edge") if min(height, width) < SIZE else image
    h, w = image.shape[:2]
    y_grid = sorted({*range(0, h - SIZE + 1, 176), h - SIZE})
    x_grid = sorted({*range(0, w - SIZE + 1, 176), w - SIZE})
    boxes, confidences = [], []
    locations = [(y, x) for y in y_grid for x in x_grid]
    model.eval()
    for i in range(0, len(locations), batch):
        points = locations[i:i + batch]
        crops = np.stack([image[y:y + SIZE, x:x + SIZE] for y, x in points])
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            heat, bbox, _ = model(inputs(crops, device))
            heat = heat.float().sigmoid()
        peaks = ((heat == F.max_pool2d(heat, 3, stride=1, padding=1)) & (heat >= threshold))[:, 0]
        heat, bbox, peaks = heat[:, 0].cpu().numpy(), bbox.float().cpu().numpy(), peaks.cpu().numpy()
        for k, (y0, x0) in enumerate(points):
            y, x = np.nonzero(peaks[k])
            bw, bh, ox, oy = bbox[k][:, y, x]
            dx, dy = (x + ox.clip(0, 1)) * STRIDE, (y + oy.clip(0, 1)) * STRIDE
            valid = ((dx >= (21 if x0 else 0)) & (dy >= (21 if y0 else 0)) &
                     (dx < (SIZE - 21 if x0 + SIZE < w else width - x0)) &
                     (dy < (SIZE - 21 if y0 + SIZE < h else height - y0)))
            bw, bh = np.exp(bw[valid].clip(-.1, 5)) * STRIDE, np.exp(bh[valid].clip(-.1, 5)) * STRIDE
            cx, cy = dx[valid] + x0, dy[valid] + y0
            bounds = np.column_stack(((cx - bw / 2).clip(0, width), (cy - bh / 2).clip(0, height),
                                      (cx + bw / 2).clip(0, width), (cy + bh / 2).clip(0, height)))
            boxes.append(bounds)
            confidences.append(heat[k, y[valid], x[valid]])
    boxes, confidences = np.concatenate(boxes), np.concatenate(confidences)
    keep = nms(boxes, confidences)
    return boxes[keep], confidences[keep]


def match(predicted, truth, threshold=.5):
    overlaps = box_iou(predicted, truth)
    # Prefer the number of valid matches before optimizing their overlap.
    cost = np.where(overlaps >= threshold, min(overlaps.shape) + 1. + overlaps, 0.)
    p, t = linear_sum_assignment(-cost)
    keep = overlaps[p, t] >= threshold
    return p[keep], t[keep]


def validate(model, records, device):
    counters = np.zeros((5, 3), np.int64)
    cutoffs = [.16, .22, .28, .36, .45]
    for record in records:
        path = SOURCE_CACHE / f"{record['chip']}.npy"
        image = (np.load(path, mmap_mode="r")[..., :3] if path.exists()
                 else read_rgb(Path(record["source"]) / "images" / f"{record['chip']}_pre_disaster.png"))
        gt = building_boxes(record)
        boxes, conf = detect(model, image, device, cutoffs[0])
        for j, threshold in enumerate(cutoffs):
            filtered = boxes[conf >= threshold]
            p, _ = match(filtered, gt)
            counters[j] += [len(p), len(filtered), len(gt)]
    scores = 2 * counters[:, 0] / np.maximum(1, counters[:, 1:].sum(1))
    j = int(np.argmax(scores))
    return {"detection_f1_50": float(scores[j]), "threshold": cutoffs[j],
            "counts": counters[j].tolist()}


def load_detector(path, device):
    state = torch.load(path, map_location=device)
    model = CenterDetector().to(device)
    model.load_state_dict(state["model"])
    return model.eval(), state


def fit(device, epochs, steps, batch):
    torch.manual_seed(59111)
    rng = np.random.default_rng(593219)
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    TARGET_CACHE.mkdir(parents=True, exist_ok=True)
    for n, row in enumerate(manifest["train"], 1):
        chip = row["chip"]
        target = TARGET_CACHE / f"{chip}.npy"
        if not target.exists():
            source = SOURCE_CACHE / f"{chip}.npy"
            if source.exists():
                shape = np.load(source, mmap_mode="r").shape[:2]
            else:
                pre = Path(row["source"]) / "images" / f"{chip}_pre_disaster.png"
                shape = read_rgb(pre).shape[:2]
            temporary = target.with_suffix(".tmp.npy")
            np.save(temporary, make_targets(building_boxes(row), shape).astype(np.float16))
            temporary.replace(target)
        if n % 25 == 0 or n == len(manifest["train"]):
            print(f"Center targets: {n}/{len(manifest['train'])}", flush=True)
    model = CenterDetector().to(device)
    base = torch.load(ROOT.parent / "run/building_best.pt", map_location=device)["model"]
    model.load_state_dict({k: v for k, v in base.items() if k in model.state_dict()}, strict=False)
    optim = torch.optim.AdamW(model.parameters(), lr=.00055, weight_decay=.02)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs, eta_min=.00007)
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    out = ROOT / "run/centers"
    out.mkdir(parents=True, exist_ok=True)
    best = -1.
    history = []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        running = 0.
        model.train()
        for _ in range(steps):
            img, targets = batch_crops(manifest["train"], rng, batch)
            rgb = inputs(img, device)
            gain = torch.rand((len(img), 3, 1, 1), device=device) * .24 + .88
            shift = torch.rand((len(img), 3, 1, 1), device=device) * .24 - .12
            rgb = (rgb * gain + shift).clamp(-1, 1)
            y = torch.from_numpy(np.ascontiguousarray(targets.transpose(0, 3, 1, 2))).to(device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = loss_for(model(rgb), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 8.)
            scaler.step(optim)
            scaler.update()
            running += float(loss.detach())
        schedule.step()
        row = {"epoch": epoch, "loss": running / steps, "elapsed": time.monotonic() - start}
        if epoch == 1 or epoch % 4 == 0 or epoch == epochs:
            row.update(validate(model, manifest["val"], device))
            if row["detection_f1_50"] > best:
                best = row["detection_f1_50"]
                temp = out / "detector.tmp"
                torch.save({"model": model.state_dict(), **row}, temp)
                temp.replace(out / "detector.pt")
        history.append(row)
        write_json(out / "history.json", history)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--epochs", type=int, default=20)
    cli.add_argument("--steps", type=int, default=96)
    cli.add_argument("--batch", type=int, default=12)
    args = cli.parse_args()
    if min(args.epochs, args.steps, args.batch) < 1:
        cli.error("epochs, steps and batch must be positive")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    fit(torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.epochs, args.steps, args.batch)
