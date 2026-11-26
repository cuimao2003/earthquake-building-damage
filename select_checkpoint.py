"""Choose and calibrate a final checkpoint using the validation split only."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluation import (Calibration, DAMAGE_THRESHOLDS, normalize_damage, predict, predict_map,
                        smooth_damage)
from infer import RUN, load_model
from prepare import CLASSES, DATA, write_json


SETTINGS = ({"views": 1, "sigma": 0.0},
            {"views": 4, "sigma": 0.0},
            {"views": 4, "sigma": 5.0},
            {"views": 4, "sigma": 5.0, "normalize": True})
FINAL_THRESHOLDS = np.concatenate(([.12, .16, .20, .24, .28, .31], DAMAGE_THRESHOLDS))


def calibrated_configs(path, records, cache, device, batch):
    model, state = load_model(path, device)
    accumulators = [Calibration(damage_thresholds=FINAL_THRESHOLDS) for _ in SETTINGS]
    amp = device.type == "cuda"
    for row in records:
        chip = np.load(cache / "earthquake" / row["file"], mmap_mode="r")
        pre, post, truth = chip[..., :3], chip[..., 3:6], chip[..., 6]
        regular = predict(model, pre, post, device, batch, amp)
        rotated = predict_map(model, pre, post, device, batch, amp, views=4)
        smoothed = smooth_damage(rotated, 5.)
        for acc, probs in zip(accumulators, (regular, rotated, smoothed, normalize_damage(smoothed))):
            acc.add(truth, probs)
    results = []
    for setting, acc in zip(SETTINGS, accumulators):
        thresholds, metrics = acc.fit()
        score = 0.25 * metrics["footprint"]["f1"] + 0.75 * metrics["damaged_macro_f1"]
        results.append({"checkpoint": str(path.resolve()), "epoch": state["epoch"],
                        "focused": state.get("focused", False), "inference": setting,
                        "thresholds": thresholds, "validation": metrics, "selection_score": score})
        print(json.dumps({"source": str(path), "epoch": state["epoch"], **setting, "score": score,
                          "footprint_f1": metrics["footprint"]["f1"],
                          "minor_f1": metrics["classes"][CLASSES[2]]["f1"],
                          "severe_f1": metrics["classes"][CLASSES[3]]["f1"]}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--cache", type=Path, default=DATA)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = json.loads((args.cache / "earthquake.json").read_text(encoding="utf-8"))["val"]
    results = [config for path in args.candidates
               for config in calibrated_configs(path, records, args.cache, device, args.batch)]
    best = max(results, key=lambda row: row["selection_score"])
    state = torch.load(best["checkpoint"], map_location="cpu")
    for name in ("inference", "thresholds", "validation", "selection_score"):
        state[name] = best[name]
    args.run.mkdir(parents=True, exist_ok=True)
    temp = args.run / "damage_model.tmp"
    torch.save(state, temp)
    temp.replace(args.run / "damage_model.pt")
    write_json(args.run / "selection.json", {"selected": best, "candidates": results})


if __name__ == "__main__":
    main()
