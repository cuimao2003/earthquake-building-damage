"""Rank the Turkey test scenes by building-level accuracy for a typical-result figure."""
import json
from pathlib import Path

import numpy as np
import torch

from classifier import predict_grade
from detector import match
from inventory import ROOT
from turkey import load_regions
from turkey_automatic import Assessment, OUT, detection_image, paired

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = Assessment(OUT, DEVICE)
    config = json.loads((OUT / "automatic_thresholds.json").read_bytes())
    thresholds = np.asarray(config["thresholds"], dtype=np.float32)
    test = json.loads((ROOT / "data/turkey/split.json").read_bytes())["splits"]["test"]
    rows = []
    for number in test:
        before, after = paired(number)
        boxes, scores = model.automatic(before, after, *detection_image(before))
        grades = predict_grade(scores, thresholds)
        annotated = load_regions(number)
        truth = np.asarray([box for box, _ in annotated], np.float32).reshape(-1, 4)
        labels = np.asarray([grade for _, grade in annotated], np.int8)
        matched, assigned = match(boxes, truth)
        correct = int((grades[matched] == labels[assigned]).sum())
        row = {
            "scene": int(number),
            "annotated": int(len(labels)),
            "predicted": int(len(boxes)),
            "matched": int(len(matched)),
            "correct": correct,
            "accuracy": correct / max(1, len(labels)),
            "class_support": np.bincount(labels, minlength=3).tolist(),
        }
        rows.append(row)
        print(f"scene {number:02d}: {correct}/{len(labels)} = {row['accuracy']:.4f} "
              f"(predicted {len(boxes)}, matched {len(matched)})", flush=True)
    rows.sort(key=lambda item: -item["accuracy"])
    (OUT / "typical_ranking.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== ranking ===")
    for row in rows:
        print(f"scene {row['scene']:02d}: accuracy {row['accuracy']:.4f} ({row['correct']}/{row['annotated']})")


if __name__ == "__main__":
    main()
