"""Evaluate head ensembles and their fusion with the reference classifier on the validation split.

Usage:
  python run_fusion_grid.py --cache data/proposal_lower_floor \
      --heads run/proposal_attention_floor24/head.pt --out run/grid_floor24.json
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import train_proposal_transformer as T
from classifier import predict_grade
from detector import match
from infer_retrain import tune_automatic_thresholds
from inventory import ROOT
from roi_transformer import RoofTransformer


def load_head(path, device):
    state = torch.load(path, map_location="cpu")
    head = RoofTransformer(state["width"], state["variant"]).to(device)
    head.load_state_dict(state["head"])
    head.eval()
    return head, state


def scores_for(heads, scene, device):
    values = [T.score(head, scene["features"], scene["roi"], device)[:, 1:] for head in heads]
    return np.mean(values, axis=0)


def evaluate(scores, targets, missed):
    return tune_automatic_thresholds(scores, targets, missed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/proposal_lower_floor")
    parser.add_argument("--heads", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weight", type=Path, default=ROOT / "weights/dinov2_vitb14_pretrain.pth")
    parser.add_argument("--detector", type=Path, default=ROOT / "run/swin_loc_mix/detector.pt")
    parser.add_argument("--reference", type=Path, default=ROOT / "run/mexico_large_selection")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = [load_head(path, device)[0] for path in args.heads]
    records = json.loads(T.MANIFEST.read_bytes())["val"]
    chips = json.loads((args.cache / "val/chips.json").read_bytes())
    if [row["chip"] for row in records] != chips:
        raise ValueError("Validation scene order differs from the locked cache")

    reference = T.SelectedGrader(args.reference, device)
    fixed, proposed, targets = [], [], []
    missed = np.zeros(3, np.int64)
    for index, (row, scene) in enumerate(zip(records, T.load_split(args.cache, "val")), 1):
        boxes, truth, grades = scene["boxes"], scene["truth"], scene["grades"]
        matched, assigned = match(boxes, truth)
        target = np.full(len(boxes), -1, np.int8)
        target[matched] = np.where(grades[assigned] >= 0, grades[assigned], -2)
        known = target >= -1
        remaining = np.ones(len(grades), bool)
        remaining[assigned] = False
        missed += np.bincount(grades[remaining & (grades >= 0)], minlength=3)
        _, baseline = reference.classify(*T.paired_images(row), boxes)
        fixed.append(baseline[known])
        proposed.append(scores_for(heads, scene, device)[known])
        targets.append(target[known])
        if index % 5 == 0 or index == len(records):
            print(f"validation scenes {index}/{len(records)}", flush=True)

    baseline, proposed, targets = map(np.concatenate, (fixed, proposed, targets))
    result = {"cache": str(args.cache.resolve()), "heads": [str(p.resolve()) for p in args.heads],
              "scenes": len(records), "head_only": None, "reference_only": None, "trials": []}

    head_only = evaluate(proposed, targets, missed)
    result["head_only"] = head_only
    print(f"head ensemble only: macro F1 {head_only['validation']['macro_f1']:.4f}", flush=True)

    reference_only = evaluate(baseline, targets, missed)
    result["reference_only"] = reference_only
    print(f"reference only: macro F1 {reference_only['validation']['macro_f1']:.4f}", flush=True)

    best = None
    for light in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        for heavy in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            weight = np.asarray([light, heavy])
            fused = T.fusion_scores(baseline, proposed, weight)
            setting = evaluate(fused, targets, missed)
            setting["weight"] = weight.tolist()
            setting["validity_threshold"] = 0.0
            result["trials"].append(setting)
            if best is None or setting["validation"]["macro_f1"] > best["validation"]["macro_f1"]:
                best = setting
    result["selected"] = best
    print(f"best fusion weight {best['weight']}, macro F1 "
          f"{best['validation']['macro_f1']:.4f}, damaged {best['validation']['damaged_macro_f1']:.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
