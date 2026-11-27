"""Audit detector test-time augmentation (flip + WBF) on the Mexico validation split.

Reports localization P/R/F1 and per-grade matched counts at several score thresholds,
plus the perfect-grading macro-F1 ceiling implied by those matches. No retraining.

Usage:
  python detector_tta_audit.py --flip h --result run/detector_tta_audit_h.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from detector import match
from infer_retrain import ground_truth, paired_images
from inventory import ROOT, write_json
from swin_locator import load_swin, swin_boxes


def wbf(boxes_a, scores_a, boxes_b, scores_b, iou_threshold=0.55):
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
    fused_boxes = np.stack([c[0] / c[1] for c in clusters]).astype(np.float32)
    fused_scores = np.asarray([c[2] for c in clusters], dtype=np.float32)
    return fused_boxes, fused_scores


def proposals(model, state, image, device):
    return swin_boxes(model, image, device, 0.12, tile=state.get("tile_size") or 256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, default=ROOT / "run/swin_loc_mix/detector.pt")
    parser.add_argument("--flip", choices=("none", "h", "hv"), default="h")
    parser.add_argument("--result", type=Path, default=ROOT / "run/detector_tta_audit.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_swin(args.detector, device)
    records = json.loads((ROOT / "data/manifest.json").read_bytes())["val"]
    thresholds = (0.12, 0.16, 0.24, 0.36)
    support = np.zeros(3, np.int64)
    names = ("base", "tta")
    variants = {name: {threshold: {"matched": np.zeros(3, np.int64), "total_matched": 0,
                                   "boxes": 0, "truth": 0} for threshold in thresholds}
                for name in names}
    for index, row in enumerate(records, 1):
        pre, _ = paired_images(row)
        truth, grades = ground_truth(row)
        support += np.bincount(grades[grades >= 0], minlength=3)
        base = proposals(model, state, pre, device)
        groups = [base]
        height, width = pre.shape[:2]
        if args.flip in ("h", "hv"):
            boxes_f, scores_f = proposals(model, state, pre[:, ::-1].copy(), device)
            boxes_f = boxes_f.copy()
            boxes_f[:, 0::2] = width - boxes_f[:, 0::2][:, ::-1]
            groups.append((boxes_f, scores_f))
        if args.flip == "hv":
            boxes_v, scores_v = proposals(model, state, pre[::-1].copy(), device)
            boxes_v = boxes_v.copy()
            boxes_v[:, 1::2] = height - boxes_v[:, 1::2][:, ::-1]
            groups.append((boxes_v, scores_v))
        tta = groups[0]
        for extra in groups[1:]:
            tta = wbf(tta[0], tta[1], extra[0], extra[1])
        for name, candidate in (("base", base), ("tta", tta)):
            candidate_boxes, candidate_scores = candidate
            for threshold in thresholds:
                selected = candidate_boxes[candidate_scores >= threshold]
                matched_all, _ = match(selected, truth)
                entry = variants[name][threshold]
                entry["boxes"] += int(len(selected))
                entry["truth"] += int(len(truth))
                entry["total_matched"] += int(len(matched_all))
                for grade in range(3):
                    mask = grades == grade
                    if mask.any():
                        count, _ = match(selected, truth[mask])
                        entry["matched"][grade] += int(len(count))
        if index % 5 == 0 or index == len(records):
            print(f"detector TTA audit {index}/{len(records)}", flush=True)

    output = {"detector": str(args.detector.resolve()), "flip": args.flip,
              "support": support.tolist(), "variants": {}}
    for name in names:
        output["variants"][name] = {}
        for threshold in thresholds:
            entry = variants[name][threshold]
            precision = entry["total_matched"] / max(1, entry["boxes"])
            recall = entry["total_matched"] / max(1, entry["truth"])
            f1 = 2 * precision * recall / max(1e-9, precision + recall)
            perfect = [2 * (m / s) / (1 + m / s) if s else 0 for m, s in zip(entry["matched"], support)]
            record = {"boxes": entry["boxes"], "matched": int(entry["total_matched"]),
                      "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                      "per_grade_matched": entry["matched"].tolist(),
                      "perfect_grading_macro_ceiling": round(float(np.mean(perfect)), 4)}
            output["variants"][name][str(threshold)] = record
            print(f"{name} @ {threshold}: {json.dumps(record)}", flush=True)
    write_json(args.result, output)


if __name__ == "__main__":
    main()
