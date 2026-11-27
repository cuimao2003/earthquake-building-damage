"""Quick experiments on using the building mask for localization (xBD val).

Variants:
1. mask-derived connected-component boxes (baseline from eval_building_unet);
2. watershed instance boxes (distance-transform seeds);
3. refined TTA proposals: tighten each box to the mask pixels inside it, drop boxes
   whose mask coverage is too low;
4. watershed instance boxes merged with TTA proposals via WBF.

Usage:
  python mask_box_experiments.py --split val --result run/mask_box_experiments_val.json
"""

import argparse
import json

import cv2
import numpy as np
import torch

import train_proposal_transformer as T
from detector import match
from eval_building_unet import infer_mask
from infer_retrain import ground_truth, paired_images
from inventory import ROOT, write_json
from train_building_unet import UNet


def wbf(boxes_a, scores_a, boxes_b, scores_b, iou_threshold=.55):
    boxes = np.concatenate((boxes_a, boxes_b), axis=0)
    scores = np.concatenate((scores_a, scores_b), axis=0)
    order = np.argsort(-scores)
    clusters = []
    for index in order:
        box, score = boxes[index], float(scores[index])
        placed = False
        for cluster in clusters:
            fused = cluster[0] / cluster[1]
            left = max(fused[0], box[0]); top = max(fused[1], box[1])
            right = min(fused[2], box[2]); bottom = min(fused[3], box[3])
            inter = max(0., right - left) * max(0., bottom - top)
            area_f = (fused[2] - fused[0]) * (fused[3] - fused[1])
            area_b = (box[2] - box[0]) * (box[3] - box[1])
            iou = inter / max(1e-6, area_f + area_b - inter)
            if iou >= iou_threshold:
                cluster[0] += box * score
                cluster[1] += score
                cluster[2] = max(cluster[2], score)
                placed = True
                break
        if not placed:
            clusters.append([box * score, score, score])
    if not clusters:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return (np.stack([c[0] / c[1] for c in clusters]).astype(np.float32),
            np.asarray([c[2] for c in clusters], np.float32))


def watershed_boxes(mask, min_area=30):
    binary = (mask > .5).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    peaks = ((distance >= cv2.dilate(distance, np.ones((3, 3), np.float32))) & (distance > 2.5)).astype(np.uint8)
    count, seeds = cv2.connectedComponents(peaks)
    markers = np.zeros((mask.shape[0], mask.shape[1]), np.int32)
    markers[binary == 0] = 1
    markers[seeds > 0] = seeds[seeds > 0] + 1
    color = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    boxes = []
    for label in range(2, count + 1):
        region = (markers == label).astype(np.uint8)
        if region.sum() < min_area:
            continue
        ys, xs = np.nonzero(region)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.asarray(boxes, np.float32)


def refine_boxes(proposals, mask, min_coverage=0.15):
    boxes = []
    for box in proposals:
        y1, y2 = max(0, int(box[1])), min(mask.shape[0], int(box[3]))
        x1, x2 = max(0, int(box[0])), min(mask.shape[1], int(box[2]))
        if y2 <= y1 or x2 <= x1:
            continue
        local = mask[y1:y2, x1:x2]
        coverage = local.mean()
        if coverage < min_coverage:
            continue
        ys, xs = np.nonzero(local > .5)
        if len(xs) < 8:
            continue
        boxes.append([x1 + xs.min(), y1 + ys.min(), x1 + xs.max() + 1, y1 + ys.max() + 1])
    return np.asarray(boxes, np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--mask-checkpoint", type=__import__("pathlib").Path,
                        default=ROOT / "run/building_unet_xbd/building_unet_xbd.pt")
    parser.add_argument("--detector", type=__import__("pathlib").Path,
                        default=ROOT / "run/swin_loc_mix/detector.pt")
    parser.add_argument("--result", type=__import__("pathlib").Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.mask_checkpoint, map_location="cpu")
    model = UNet().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    detector, detector_state = T.load_swin(args.detector, device)
    records = json.loads(T.MANIFEST.read_bytes())[args.split]
    support = np.zeros(3, np.int64)
    variants = {}
    for name in ("components", "watershed", "refined_015", "refined_030", "ws_plus_tta"):
        variants[name] = {"matched": np.zeros(3, np.int64), "boxes": 0, "truth": 0, "total": 0}
    for index, row in enumerate(records, 1):
        pre, _ = paired_images(row)
        truth, grades = ground_truth(row)
        support += np.bincount(grades[grades >= 0], minlength=3)
        mask = infer_mask(model, pre, device)
        proposals = T.detector_boxes(detector, detector_state, pre, device, .36, "h")
        groups = {
            "components": _component_boxes(mask),
            "watershed": watershed_boxes(mask),
            "refined_015": refine_boxes(proposals, mask, .15),
            "refined_030": refine_boxes(proposals, mask, .30),
        }
        merged = wbf(proposals, np.ones(len(proposals), np.float32),
                     watershed_boxes(mask), np.ones(len(watershed_boxes(mask)), np.float32) * .5)
        groups["ws_plus_tta"] = merged[0]
        for name, boxes in groups.items():
            entry = variants[name]
            matched, _ = match(boxes, truth)
            entry["boxes"] += len(boxes)
            entry["truth"] += len(truth)
            entry["total"] += len(matched)
            for grade in range(3):
                grade_mask = grades == grade
                if grade_mask.any():
                    count, _ = match(boxes, truth[grade_mask])
                    entry["matched"][grade] += int(len(count))
        if index % 5 == 0 or index == len(records):
            print(f"mask box experiments {index}/{len(records)}", flush=True)
    output = {"split": args.split, "support": support.tolist(), "variants": {}}
    for name, entry in variants.items():
        precision = entry["total"] / max(1, entry["boxes"])
        recall = entry["total"] / max(1, entry["truth"])
        perfect = [2 * (m / s) / (1 + m / s) if s else 0 for m, s in zip(entry["matched"], support)]
        output["variants"][name] = {
            "boxes": entry["boxes"], "matched": entry["total"],
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
            "per_grade_matched": entry["matched"].tolist(),
            "perfect_grading_macro_ceiling": round(float(np.mean(perfect)), 4),
        }
        print(name, json.dumps(output["variants"][name]), flush=True)
    write_json(args.result, output)


def _component_boxes(mask, min_area=30):
    binary = (mask > .5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area or w < 3 or h < 3:
            continue
        boxes.append([x, y, x + w, y + h])
    return np.asarray(boxes, np.float32)


if __name__ == "__main__":
    main()
