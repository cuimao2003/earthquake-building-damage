"""Apply the frozen Mexico classifiers to the labeled Turkey holdout."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from classifier import matrix, predict_grade, report
from detector import match
from features import align, read_rgb
from infer_retrain import comparison, grade_metrics
from inventory import ROOT, write_json
from train_proposal_transformer import DETECTOR, WEIGHTS, ProposalGrader, detector_boxes, digest, fusion_scores
from turkey import DATA, load_regions
from turkey_automatic import DETECTION_SCALE, detection_image, rename_classes


MODELS = ("mexico_baseline", "mexico_fusion")


def labels(grader, pre, post, boxes):
    baseline, reference = grader.reference.classify(pre, post, boxes)
    if not len(boxes):
        return baseline, baseline, np.zeros(0, dtype=bool)
    new_scores = grader.raw_scores(pre, post, boxes)
    scores = fusion_scores(reference, new_scores[:, 1:], grader.fusion_weight)
    fusion = predict_grade(scores, np.asarray(grader.settings["thresholds"]))
    return baseline, fusion, new_scores[:, 0] >= grader.settings["validity_threshold"]


class Result:
    def __init__(self):
        self.confusion = np.zeros((3, 3), dtype=np.int64)
        self.manual = self.confusion.copy()
        self.missed = np.zeros(3, dtype=np.int64)
        self.extras = self.missed.copy()
        self.rows = []

    def add(self, number, truth, grades, boxes, predicted, manual):
        p, t = match(boxes, truth)
        missing = np.ones(len(truth), dtype=bool)
        extra = np.ones(len(boxes), dtype=bool)
        missing[t] = extra[p] = False
        matched_grades = matrix(grades[t], predicted[p])
        manual_grades = matrix(grades, manual)
        self.confusion += matched_grades
        self.manual += manual_grades
        self.missed += np.bincount(grades[missing], minlength=3)
        self.extras += np.bincount(predicted[extra], minlength=3)
        self.rows.append({"scene": number, "annotated": len(truth), "predicted": len(boxes),
                          "matched": len(p), "correct_location_and_grade": int(matched_grades.trace()),
                          "correct_grade_given_human_boxes": int(manual_grades.trace())})

    def summary(self, destination):
        classification = grade_metrics(self.confusion, self.missed, self.extras)
        given_boxes = report(self.manual)
        rename_classes(classification)
        rename_classes(given_boxes)
        matched, predicted, annotated = (sum(r[k] for r in self.rows) for k in ("matched", "predicted", "annotated"))
        with destination.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return {"localization": {"matched": matched, "predicted": predicted, "annotated": annotated,
                                 "precision": matched / max(1, predicted), "recall": matched / max(1, annotated),
                                 "f1": 2 * matched / max(1, predicted + annotated)},
                "correct_location_and_grade": int(self.confusion.trace()),
                "end_to_end": classification, "given_human_boxes": given_boxes}


def run(args, device):
    splits = json.loads((DATA / "split.json").read_bytes())["splits"]
    numbers = splits["test"]
    if args.result.exists() and any(args.result.iterdir()):
        raise ValueError("Use a fresh result directory")
    args.result.mkdir(parents=True, exist_ok=True)
    grader = ProposalGrader(args, device)
    results = {name: Result() for name in MODELS}
    for i, number in enumerate(numbers, 1):
        paths = [DATA / kind / f"{number:02d}.jpg" for kind in ("pre", "post")]
        images = [read_rgb(path) for path in paths]
        if images[0].shape != images[1].shape:
            raise ValueError(f"Turkey image pair {number} uses different grids")
        pre, scale = detection_image(images[0])
        post, _ = detection_image(images[1])
        del images
        annotations = load_regions(number)
        truth = np.asarray([box for box, _ in annotations], dtype=np.float32).reshape(-1, 4) * scale
        grades = np.asarray([grade for _, grade in annotations], dtype=np.int64)
        boxes = detector_boxes(grader.detector, grader.detector_state, pre, device, grader.detector_floor)
        baseline, fusion, keep = labels(grader, pre, post, boxes)
        human_baseline, human_fusion, _ = labels(grader, pre, post, truth)
        predictions = ((boxes, baseline, human_baseline), (boxes[keep], fusion[keep], human_fusion))
        for name, (selected_boxes, selected_grades, manual) in zip(MODELS, predictions):
            results[name].add(number, truth, grades, selected_boxes, selected_grades, manual)
        if i == 1:
            background = align(pre, post, grader.registration)[0]
            comparison(pre, background, truth, grades, boxes[keep], fusion[keep], args.result / "comparison.png")
        print(f"Turkey transfer {i}/{len(numbers)} | scene {number:02d} | {len(truth)} annotated", flush=True)
    summary = {name: results[name].summary(args.result / f"{name}_per_scene.csv") for name in MODELS}
    summary.update({"scenes": numbers, "match_iou": .5, "image_scale": DETECTION_SCALE,
                    "class_mapping": {"no_damage": "no visible upper damage", "minor": "partial",
                                      "major_or_destroyed": "collapsed"},
                    "mexico_baseline_thresholds": grader.reference.automatic_thresholds.tolist(),
                    "mexico_fusion_thresholds": grader.settings["thresholds"],
                    "detector_threshold": grader.detector_state["threshold"],
                    "fusion_sha256": digest(args.fusion)})
    write_json(args.result / "test.json", summary)
    for name in MODELS:
        row = summary[name]
        print(f"{name} | end-to-end accuracy {row['end_to_end']['accuracy']:.4f} | "
              f"macro F1 {row['end_to_end']['macro_f1']:.4f} | "
              f"with human boxes {row['given_human_boxes']['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run/proposal_attention")
    parser.add_argument("--fusion", type=Path, default=ROOT / "run/proposal_fusion/fusion.json")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--detector", type=Path, default=DETECTOR)
    parser.add_argument("--encode-batch", type=int, default=24)
    parser.add_argument("--result", type=Path, default=ROOT / "run/turkey_mexico_transfer")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    run(args, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
