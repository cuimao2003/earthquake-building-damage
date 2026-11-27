"""Given-human-box grading diagnostic for a proposal Transformer head.

Encodes the human annotation boxes of the validation split with the frozen DINOv2
encoder and scores them with the trained head. Reports three measurements:
1. automatic-mode thresholds (what the end-to-end model uses);
2. thresholds re-calibrated on the given boxes (upper bound, in-sample);
3. leave-one-scene-out calibrated thresholds (honest estimate).
This isolates grading quality from localization.

Usage:
  python given_box_diagnostic.py --out run/mix_palu_pooled --result run/given_box_mix_pooled.json
"""

import argparse
import json

import numpy as np
import torch

import train_proposal_transformer as T
from classifier import calibrate, predict_grade
from infer_retrain import grade_metrics, paired_images
from inventory import ROOT, write_json
from roi_transformer import RoofTransformer, encode_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=__import__("pathlib").Path, required=True)
    parser.add_argument("--weights", type=__import__("pathlib").Path, default=ROOT / "weights/dinov2_vitb14_pretrain.pth")
    parser.add_argument("--encode-batch", type=int, default=24)
    parser.add_argument("--result", type=__import__("pathlib").Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.out / "head.pt", map_location="cpu")
    head = RoofTransformer(checkpoint["width"], checkpoint["variant"]).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    automatic = np.asarray(json.loads((args.out / "validation.json").read_bytes())["thresholds"], np.float32)
    encoder = T.load_encoder(args.weights, device)
    records = json.loads(T.MANIFEST.read_bytes())["val"]
    scene_scores, scene_grades = [], []
    for index, row in enumerate(records, 1):
        before, after = paired_images(row)
        truth_boxes, grades = T.ground_truth(row)
        known = grades >= 0
        boxes = truth_boxes[known]
        registered = T.align(before, after, "keypoint")[0]
        features, roi = encode_boxes(encoder, before, registered, boxes, device, args.encode_batch)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(features), 128):
                tokens = torch.from_numpy(features[start:start + 128]).to(device)
                weights = torch.from_numpy(roi[start:start + 128]).to(device).float()
                probability = head(tokens, weights).float().sigmoid()
                damage, severe = probability[:, 1], probability[:, 2]
                three = torch.stack((1 - damage, damage * (1 - severe), damage * severe), dim=1)
                outputs.append(three.cpu().numpy())
        scene_scores.append(np.concatenate(outputs))
        scene_grades.append(grades[known])
        if index % 5 == 0 or index == len(records):
            print(f"given-box validation {index}/{len(records)}", flush=True)
    scores = np.concatenate(scene_scores)
    grades = np.concatenate(scene_grades)

    def metrics_with(thresholds):
        predicted = predict_grade(scores, np.asarray(thresholds))
        table = np.zeros((3, 3), np.int64)
        np.add.at(table, (grades, predicted), 1)
        return grade_metrics(table, np.zeros(3, np.int64), np.zeros(3, np.int64)), table

    auto_metrics, auto_table = metrics_with(automatic)
    tuned, tuned_metrics = calibrate(scores, grades)
    tuned_report, tuned_table = metrics_with(tuned)

    loso_table = np.zeros((3, 3), np.int64)
    for index in range(len(scene_scores)):
        train_scores = np.concatenate([scene_scores[i] for i in range(len(scene_scores)) if i != index])
        train_grades = np.concatenate([scene_grades[i] for i in range(len(scene_grades)) if i != index])
        choice, _ = calibrate(train_scores, train_grades)
        predicted = predict_grade(scene_scores[index], np.asarray(choice))
        np.add.at(loso_table, (scene_grades[index], predicted), 1)
    loso_metrics = grade_metrics(loso_table, np.zeros(3, np.int64), np.zeros(3, np.int64))

    result = {
        "head": str((args.out / "head.pt").resolve()),
        "boxes": int(len(grades)),
        "automatic_thresholds": automatic.tolist(),
        "automatic_mode": {"metrics": auto_metrics, "matrix": auto_table.tolist()},
        "given_box_tuned_thresholds": list(map(float, tuned)),
        "given_box_tuned": {"metrics": tuned_report, "matrix": tuned_table.tolist()},
        "loso": {"metrics": loso_metrics, "matrix": loso_table.tolist()},
    }
    write_json(args.result, result)
    summary = {
        "boxes": int(len(grades)),
        "automatic_macro": round(auto_metrics["macro_f1"], 4),
        "tuned_macro": round(tuned_report["macro_f1"], 4),
        "tuned_damaged": round(tuned_report["damaged_macro_f1"], 4),
        "loso_macro": round(loso_metrics["macro_f1"], 4),
        "loso_damaged": round(loso_metrics["damaged_macro_f1"], 4),
        "tuned": {k: {"f1": round(v["f1"], 4), "support": v["support"]}
                  for k, v in tuned_report["classes"].items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
