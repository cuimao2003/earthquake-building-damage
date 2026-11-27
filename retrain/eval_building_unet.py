"""Evaluate the building UNet on xBD Mexico validation and use its mask to filter TTA proposals.

Reports:
1. mask pixel Dice/IoU vs polygons rasterized from the xBD pre-disaster labels;
2. localization P/R/F1 from mask-derived component boxes;
3. localization P/R/F1 + perfect-grading ceiling after filtering flip-TTA proposals by
   the mask probability inside each candidate box.

Usage:
  python eval_building_unet.py --split val --checkpoint run/building_unet/building_unet.pt \
      --result run/building_unet/xbd_val_audit.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely import wkt
from torch.nn import functional as F

import train_proposal_transformer as T
from detector import match
from infer_retrain import ground_truth, paired_images
from inventory import ROOT, write_json
from train_building_unet import UNet

WINDOW = 256
STRIDE = 128


def infer_mask(model, image, device, batch=8):
    height, width = image.shape[:2]
    total = np.zeros((height, width), np.float32)
    count = np.zeros((height, width), np.float32)
    patches, positions = [], []
    for top in range(0, max(1, height - WINDOW + 1), STRIDE):
        for left in range(0, max(1, width - WINDOW + 1), STRIDE):
            top = min(top, max(0, height - WINDOW))
            left = min(left, max(0, width - WINDOW))
            patch = image[top:top + WINDOW, left:left + WINDOW].astype(np.float32) / 255.
            patch = (patch - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
            patches.append(torch.from_numpy(patch.transpose(2, 0, 1)))
            positions.append((top, left))
            if len(patches) == batch:
                yield_buffer = None
                probs = run_batch(model, patches, device)
                for (t, l), prob in zip(positions, probs):
                    total[t:t + WINDOW, l:l + WINDOW] += prob
                    count[t:t + WINDOW, l:l + WINDOW] += 1
                patches, positions = [], []
    if patches:
        probs = run_batch(model, patches, device)
        for (t, l), prob in zip(positions, probs):
            total[t:t + WINDOW, l:l + WINDOW] += prob
            count[t:t + WINDOW, l:l + WINDOW] += 1
    return total / np.maximum(count, 1.)


@torch.no_grad()
def run_batch(model, patches, device, size=WINDOW):
    tensor = torch.stack(patches).to(device)
    logits = model(tensor)
    if logits.shape[-1] != size:
        logits = F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
    return logits[:, 0].sigmoid().cpu().numpy()


def gt_mask(row):
    root = Path(row["source"])
    labels = json.loads((root / "labels" / f"{row['chip']}_pre_disaster.json").read_bytes())["features"]["xy"]
    sheet = np.zeros((1024, 1024), np.uint8)
    for item in labels:
        geometry = wkt.loads(item["wkt"])
        coords = np.asarray(geometry.exterior.coords, np.float32)
        cv2.fillPoly(sheet, [np.round(coords).astype(np.int32)], 1)
    return sheet.astype(bool)


def component_boxes(mask, min_area=30):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area or w < 3 or h < 3:
            continue
        boxes.append([x, y, x + w, y + h])
    return np.asarray(boxes, np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "run/building_unet/building_unet.pt")
    parser.add_argument("--detector", type=Path, default=ROOT / "run/swin_loc_mix/detector.pt")
    parser.add_argument("--result", type=Path, default=ROOT / "run/building_unet/xbd_audit.json")
    parser.add_argument("--tta", choices=("none", "h"), default="h")
    parser.add_argument("--threshold", type=float, default=0.36)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu")
    model = UNet().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    detector, detector_state = T.load_swin(args.detector, device)
    records = json.loads(T.MANIFEST.read_bytes())[args.split]
    dice, iou = [], []
    detection = {"mask_boxes": {"boxes": 0, "matched": 0, "truth": 0},
                 "bbox": {}, "filtered": {}}
    for index, row in enumerate(records, 1):
        pre, _ = paired_images(row)
        truth, grades = ground_truth(row)
        mask = infer_mask(model, pre, device)
        target = gt_mask(row)
        probability = mask > .5
        intersection = (probability & target).sum()
        union = (probability | target).sum()
        dice.append(2 * intersection / max(1, probability.sum() + target.sum()))
        iou.append(intersection / max(1, union))
        boxes = component_boxes(probability)
        matched, _ = match(boxes, truth)
        detection["mask_boxes"]["boxes"] += len(boxes)
        detection["mask_boxes"]["matched"] += len(matched)
        detection["mask_boxes"]["truth"] += len(truth)
        proposals = T.detector_boxes(detector, detector_state, pre, device, args.threshold, args.tta)
        coverage = np.asarray([mask[int(b[1]):int(b[3]), int(b[0]):int(b[2])].mean() if
                               (int(b[2]) > int(b[0]) and int(b[3]) > int(b[1])) else 0.
                               for b in proposals], np.float32)
        entry = detection["bbox"].setdefault("all", {"boxes": 0, "matched": 0, "truth": 0,
                                                     "per_grade": [0, 0, 0]})
        matched_all, assigned = match(proposals, truth)
        entry["boxes"] += len(proposals)
        entry["matched"] += len(matched_all)
        entry["truth"] += len(truth)
        for grade in range(3):
            grade_mask = grades == grade
            if grade_mask.any():
                count, _ = match(proposals, truth[grade_mask])
                entry["per_grade"][grade] += int(len(count))
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            keep = coverage >= threshold
            selected = proposals[keep]
            matched, _ = match(selected, truth)
            bucket = detection["filtered"].setdefault(str(threshold), {"boxes": 0, "matched": 0, "truth": 0,
                                                                       "per_grade": [0, 0, 0]})
            bucket["boxes"] += len(selected)
            bucket["matched"] += len(matched)
            bucket["truth"] += len(truth)
            for grade in range(3):
                grade_mask = grades == grade
                if grade_mask.any():
                    count, _ = match(selected, truth[grade_mask])
                    bucket["per_grade"][grade] += int(len(count))
        if index % 5 == 0 or index == len(records):
            print(f"xBD mask audit {index}/{len(records)}", flush=True)

    support = np.zeros(3, np.int64)
    for row in records:
        _, grades = ground_truth(row)
        support += np.bincount(grades[grades >= 0], minlength=3)
    output = {"split": args.split, "scenes": len(records), "checkpoint": str(args.checkpoint.resolve()),
              "mask_dice": round(float(np.mean(dice)), 4), "mask_iou": round(float(np.mean(iou)), 4),
              "support": support.tolist(), "detection": {}}
    for name, bucket in (("mask_boxes", detection["mask_boxes"]), ("bbox_all", detection["bbox"]["all"])):
        precision = bucket["matched"] / max(1, bucket["boxes"])
        recall = bucket["matched"] / max(1, bucket["truth"])
        output["detection"][name] = {
            "boxes": bucket["boxes"], "matched": bucket["matched"],
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
            "per_grade_matched": bucket.get("per_grade", [0, 0, 0]),
        }
    for threshold, bucket in detection["filtered"].items():
        precision = bucket["matched"] / max(1, bucket["boxes"])
        recall = bucket["matched"] / max(1, bucket["truth"])
        perfect = [2 * (m / s) / (1 + m / s) if s else 0 for m, s in zip(bucket["per_grade"], support)]
        output["detection"][f"filtered_{threshold}"] = {
            "boxes": bucket["boxes"], "matched": bucket["matched"],
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
            "per_grade_matched": bucket["per_grade"],
            "perfect_grading_macro_ceiling": round(float(np.mean(perfect)), 4),
        }
    write_json(args.result, output)
    print(json.dumps({k: output[k] for k in ("mask_dice", "mask_iou", "detection")},
                     ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
