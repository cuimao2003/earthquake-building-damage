"""Scene-level bootstrap confidence intervals for a proposal head's validation metrics.

The 27-scene validation set has only 37 light and 10 heavy buildings, so point
estimates of macro F1 are noisy. This script recomputes per-scene confusion with the
locked thresholds, then resamples scenes with replacement to report the 95% interval.

Usage:
  python bootstrap_validation.py --cache data/proposal_palu_mix --out run/mix_palu_pooled \
      --result run/bootstrap_mix_palu_pooled.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from classifier import predict_grade
from detector import match
from infer_retrain import grade_metrics
from inventory import ROOT, write_json
from roi_transformer import RoofTransformer


def load_head(path, device):
    state = torch.load(path, map_location="cpu")
    head = RoofTransformer(state["width"], state["variant"]).to(device)
    head.load_state_dict(state["head"])
    head.eval()
    return head


def scene_confusion(head, scene, thresholds, validity, device):
    with torch.no_grad():
        features = torch.from_numpy(scene["features"]).to(device)
        roi = torch.from_numpy(scene["roi"]).to(device).float()
        outputs = []
        for start in range(0, len(features), 128):
            logits = head(features[start:start + 128], roi[start:start + 128])
            probabilities = logits.float().sigmoid()
            damage, severe = probabilities[:, 1], probabilities[:, 2]
            grades = torch.stack((1 - damage, damage * (1 - severe), damage * severe), dim=1)
            outputs.append(torch.cat((probabilities[:, :1], grades), dim=1))
        scores = torch.cat(outputs).cpu().numpy()
    boxes, truth, grades = scene["boxes"], scene["truth"], scene["grades"]
    keep = scores[:, 0] >= validity
    boxes = boxes[keep]
    estimates = predict_grade(scores[keep, 1:], np.asarray(thresholds))
    matched, assigned = match(boxes, truth)
    matrix = np.zeros((3, 3), np.int64)
    np.add.at(matrix, (grades[assigned], estimates[matched]), 1)
    remaining = np.ones(len(grades), bool)
    remaining[assigned] = False
    missed = np.bincount(grades[remaining & (grades >= 0)], minlength=3)
    predicted = np.ones(len(boxes), bool)
    predicted[matched] = False
    extra = np.bincount(estimates[predicted], minlength=3)
    return matrix, missed, extra


def aggregate(parts):
    matrix = np.sum([part[0] for part in parts], axis=0)
    missed = np.sum([part[1] for part in parts], axis=0)
    extra = np.sum([part[2] for part in parts], axis=0)
    metrics = grade_metrics(matrix, missed, extra)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/proposal_palu_mix")
    parser.add_argument("--out", type=Path, required=True, help="trained head run directory")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration = json.loads((args.out / "validation.json").read_bytes())
    thresholds = calibration["thresholds"]
    validity = calibration.get("validity_threshold", 0.0)
    head = load_head(args.out / "head.pt", device)
    records = json.loads((ROOT / "data/manifest.json").read_bytes())
    chips = json.loads((args.cache / "val/chips.json").read_bytes())
    from train_proposal_transformer import load_split
    scenes = list(load_split(args.cache, "val"))
    if len(scenes) != len(chips):
        raise ValueError("cache validation split mismatch")
    parts = [scene_confusion(head, scene, thresholds, validity, device) for scene in scenes]
    full = aggregate(parts)
    rng = np.random.default_rng(args.seed)
    values = {"macro_f1": [], "damaged_macro_f1": [], "accuracy": []}
    for _ in range(args.rounds):
        picks = rng.integers(0, len(parts), len(parts))
        metrics = aggregate([parts[index] for index in picks])
        for key in values:
            values[key].append(metrics[key])
    summary = {"out": str(args.out.resolve()), "scenes": len(parts), "rounds": args.rounds,
               "point": {key: full[key] for key in ("macro_f1", "damaged_macro_f1", "accuracy")}}
    for key, samples in values.items():
        array = np.asarray(samples)
        summary[key] = {"mean": float(array.mean()), "low95": float(np.percentile(array, 2.5)),
                        "high95": float(np.percentile(array, 97.5))}
    write_json(args.result, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
