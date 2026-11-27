"""TTA localization + old classifier, with thresholds recalibrated to the TTA box distribution.

Fixes the earlier hybrid failure mode (old classifier thresholds were calibrated on the
old localizer's boxes). Val is used to calibrate the two grade thresholds; the 34-scene
holdout is then evaluated once with those thresholds.

Usage:
  python hybrid_tta_calibrated.py --result run/hybrid_tta_calibrated.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_proposal_transformer as T
from classifier import odds, predict_grade
from detector import match
from infer_retrain import SelectedGrader, grade_metrics, ground_truth, paired_images
from inventory import ROOT, write_json

QUANTILES = [0.25, .40, .55, .70, .80, .88, .92, .95, .97, .98, .99, .995, .998, 1.]


def collect(records, grader, detector, detector_state, device, threshold, tta):
    table = np.zeros((3, 3), np.int64)
    missed = np.zeros(3, np.int64)
    scores, matched_grades, unmatched_mask = [], [], []
    for index, row in enumerate(records, 1):
        pre, post = paired_images(row)
        truth, grades = ground_truth(row)
        boxes = T.detector_boxes(detector, detector_state, pre, device, threshold, tta)
        estimates, probs = grader.classify(pre, post, boxes)
        if len(boxes):
            matched, assigned = match(boxes, truth)
            assigned_mask = np.ones(len(truth), bool)
            assigned_mask[assigned] = False
            keep = grades[assigned] >= 0
            np.add.at(table, (grades[assigned[keep]], estimates[matched[keep]]), 1)
            missed += np.bincount(grades[assigned_mask & (grades >= 0)], minlength=3)
            matched_mask = np.zeros(len(boxes), bool)
            matched_mask[matched] = True
            scores.append(probs)
            matched_grades.append(grades[assigned[keep]])
            position = np.zeros(len(boxes), np.int64)
            position[matched[keep]] = np.arange(len(matched_grades[-1]))
            unmatched_mask.append(np.isin(np.arange(len(boxes)), matched[keep]))
        if index % 8 == 0 or index == len(records):
            print(f"collect {index}/{len(records)}", flush=True)
    return np.concatenate(scores), np.concatenate(matched_grades), np.concatenate(unmatched_mask), missed


def calibrate(scores, grades, matched_mask, missed):
    margins = odds(scores)
    grids = [np.unique(np.r_[-18., np.quantile(margins[:, i], QUANTILES), 18.]) for i in range(2)]
    actual = grades
    best = None
    for light in grids[0]:
        for severe in grids[1]:
            predicted = predict_grade(scores, np.array([light, severe]))
            table = np.bincount(3 * actual + predicted[matched_mask], minlength=9).reshape(3, 3)
            extras = np.bincount(predicted[~matched_mask], minlength=3)
            result = grade_metrics(table, missed, extras)
            if best is None or result["macro_f1"] > best["validation"]["macro_f1"]:
                best = {"thresholds": [float(light), float(severe)], "validation": result}
    return best


def evaluate_with(scores, grades, matched_mask, missed, thresholds):
    predicted = predict_grade(scores, np.asarray(thresholds))
    table = np.bincount(3 * grades + predicted[matched_mask], minlength=9).reshape(3, 3)
    extras = np.bincount(predicted[~matched_mask], minlength=3)
    return grade_metrics(table, missed, extras), table, extras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, default=ROOT / "run/swin_loc_mix/detector.pt")
    parser.add_argument("--classifier", type=Path, default=ROOT / "run/mexico_large_selection")
    parser.add_argument("--threshold", type=float, default=0.36)
    parser.add_argument("--tta", choices=("none", "h"), default="h")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, detector_state = T.load_swin(args.detector, device)
    grader = SelectedGrader(args.classifier, device)
    manifest = json.loads(T.MANIFEST.read_bytes())
    print("collecting validation scores", flush=True)
    val_scores, val_grades, val_unmatched, val_missed = collect(
        manifest["val"], grader, detector, detector_state, device, args.threshold, args.tta)
    best = calibrate(val_scores, val_grades, val_unmatched, val_missed)
    print("val calibrated:", json.dumps({"thresholds": best["thresholds"],
                                         "macro_f1": best["validation"]["macro_f1"],
                                         "damaged_macro_f1": best["validation"]["damaged_macro_f1"]}))
    print("collecting holdout scores", flush=True)
    test_scores, test_grades, test_unmatched, test_missed = collect(
        manifest["test"], grader, detector, detector_state, device, args.threshold, args.tta)
    holdout, table, extras = evaluate_with(test_scores, test_grades, test_unmatched, test_missed,
                                           best["thresholds"])
    output = {"detector": str(args.detector.resolve()), "classifier": str(args.classifier.resolve()),
              "threshold": args.threshold, "tta": args.tta,
              "val": {"thresholds": best["thresholds"], "validation": best["validation"]},
              "holdout": {"validation": holdout, "matrix": table.tolist(), "extras": extras.tolist()}}
    write_json(args.result, output)
    print(json.dumps({"val_macro": best["validation"]["macro_f1"],
                      "val_damaged": best["validation"]["damaged_macro_f1"],
                      "holdout_macro": holdout["macro_f1"],
                      "holdout_damaged": holdout["damaged_macro_f1"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
