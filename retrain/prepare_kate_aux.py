"""Prepare cached DINOv2 features for the KATE-CD Turkey 2023 auxiliary damage signal.

KATE-CD rows are 512x512 pre/post crops with a binary damage mask. We sample
168x168-context patches around damaged and intact locations, encode them with the
frozen DINOv2 encoder, and dump features/roi/target for the auxiliary BCE loss.

Usage:
  python prepare_kate_aux.py --split train --out data/kate_aux.npz
"""

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch

from dino import load_dino
from train_proposal_transformer import WEIGHTS as ENCODER_WEIGHTS
from inventory import ROOT
from roi_transformer import encode_boxes

KATE = Path(r"E:/datasets/kate-cd")
BOX = 64.0  # local context box; patch_pair expands it by 1.8x


def decode(row, key):
    payload = row[key]["bytes"]
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable {key}")
    return np.ascontiguousarray(image[:, :, ::-1])


def label_mask(row):
    payload = row["label"]["bytes"]
    mask = np.asarray(Image.open(io.BytesIO(payload)).convert("L"))
    return mask > 127


def sample_boxes(mask, rows, rng, positives=2, negatives=2):
    height, width = mask.shape
    damaged = np.argwhere(mask)
    boxes, targets = [], []
    if len(damaged):
        picks = rng.choice(len(damaged), size=min(positives, len(damaged)), replace=False)
        for index in picks:
            cy, cx = damaged[index]
            boxes.append([cx - BOX / 2, cy - BOX / 2, cx + BOX / 2, cy + BOX / 2])
            targets.append(1.0)
    for _ in range(negatives):
        for _try in range(20):
            cx = rng.uniform(BOX, width - BOX)
            cy = rng.uniform(BOX, height - BOX)
            patch = mask[int(cy - BOX):int(cy + BOX), int(cx - BOX):int(cx + BOX)]
            if patch.size and not patch.any():
                boxes.append([cx - BOX / 2, cy - BOX / 2, cx + BOX / 2, cy + BOX / 2])
                targets.append(0.0)
                break
    boxes = np.asarray(boxes, np.float32)
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
    return boxes, np.asarray(targets, np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--out", type=Path, default=ROOT / "data/kate_aux.npz")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    table = pq.read_table(KATE / f"{args.split}-00000-of-00001.parquet").to_pylist()
    encoder = load_dino(ENCODER_WEIGHTS, device)
    rng = np.random.default_rng(20261127)
    features, rois, targets = [], [], []
    for index, row in enumerate(table, 1):
        before, after = decode(row, "pre_image"), decode(row, "post_image")
        mask = label_mask(row)
        boxes, values = sample_boxes(mask, None, rng)
        if not len(boxes):
            continue
        visual, roi = encode_boxes(encoder, before, after, boxes, device, 24)
        features.append(visual)
        rois.append(roi)
        targets.append(values)
        if index % 50 == 0 or index == len(table):
            print(f"KATE auxiliary {args.split}: {index}/{len(table)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, features=np.concatenate(features), roi=np.concatenate(rois),
             target=np.concatenate(targets))
    summary = {"split": args.split, "rows": len(table), "patches": int(sum(len(v) for v in targets)),
               "positive": int(np.concatenate(targets).sum()), "source": str(KATE)}
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
