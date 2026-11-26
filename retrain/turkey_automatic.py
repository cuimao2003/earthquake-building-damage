"""Train and assess automatic building localization on the Turkey earthquake."""

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.ops import nms

from classifier import QUANTILES, predict_grade
from detector import match
from features import read_rgb
from finetune_turkey import ImagePatches, load, predict
from infer_retrain import COLORS, grade_metrics, tune_automatic_thresholds
from inventory import ROOT, write_json
from swin_locator import load_swin, swin_boxes, train
from turkey import DATA, load_regions


OUT = ROOT / "run/turkey_automatic"
CLASSES = ("A_no_visible_upper_damage", "B_partial", "C_collapsed")
DETECTION_SCALE = .18
SCALES = (DETECTION_SCALE, .14, .23)


def detection_image(image, resize=DETECTION_SCALE):
    height, width = image.shape[:2]
    size = (max(1, round(width * resize)), max(1, round(height * resize)))
    compact = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return compact, np.float32([size[0] / width, size[1] / height] * 2)


@lru_cache(maxsize=70)
def scene(number):
    return detection_image(read_rgb(DATA / "pre" / f"{number:02d}.jpg"))


def training_records(numbers):
    rows = []
    for number in numbers:
        boxes = np.asarray([box for box, _ in load_regions(number)], dtype=np.float32).reshape(-1, 4)
        rows.append((number, boxes * scene(number)[1]))
    return rows


def localization(model, compact, scale, threshold, device):
    boxes, confidence = swin_boxes(model, compact, device, threshold)
    return boxes / scale, confidence


def merge_detections(groups):
    boxes, scores = map(np.concatenate, zip(*groups))
    keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), .48).numpy()
    return boxes[keep], scores[keep]


class Assessment:
    def __init__(self, out, device):
        self.device = device
        self.detector, checkpoint = load_swin(out / "detector.pt", device)
        self.loc = {"threshold": checkpoint["threshold"], "scales": [DETECTION_SCALE]}
        config = out / "automatic_thresholds.json"
        if config.is_file():
            self.loc = json.loads(config.read_bytes())["localization"]
        settings = json.loads((ROOT / "run/turkey_vit/selection.json").read_bytes())["selected"]
        self.members = [(member, *load(Path(member["path"]), device)) for member in settings["models"]]

    def scores(self, pre, post, boxes):
        result = np.zeros((len(boxes), 3), dtype=np.float32)
        if not len(boxes):
            return result
        samples = {}
        for member, encoder, head, state in self.members:
            size = state.get("patch_size", 112)
            if size not in samples:
                samples[size] = ImagePatches(pre, post, boxes, size)
            result += member["weight"] * predict(encoder, head, samples[size], self.device,
                                                  batch=40 if size > 112 else 72,
                                                  tta=member["tta"],
                                                  local_evidence=state.get("local_evidence", False))
        return result

    def automatic(self, pre, post, compact, scale):
        if pre.shape != post.shape:
            raise ValueError("Pre-event and post-event imagery must use the same grid")
        predictions = []
        for resize in self.loc["scales"]:
            image, ratio = (compact, scale) if resize == DETECTION_SCALE else detection_image(pre, resize)
            predictions.append(localization(self.detector, image, ratio, self.loc["threshold"], self.device))
        boxes, _ = predictions[0] if len(predictions) == 1 else merge_detections(predictions)
        height, width = pre.shape[:2]
        boxes = boxes.clip(0, [width, height] * 2).astype(np.float32)
        boxes = boxes[((boxes[:, 2:] - boxes[:, :2]) >= 1).all(1)]
        return boxes, self.scores(pre, post, boxes)


def paired(number):
    return [read_rgb(DATA / kind / f"{number:02d}.jpg") for kind in ("pre", "post")]


def choose_localization(model, numbers):
    cutoffs = sorted({.26, .32, .38, .44, .50, .56, .62, model.loc["threshold"]})
    options = [(0,), (1,), (2,), (0, 1), (0, 2), (0, 1, 2)]
    totals = np.zeros((len(options), len(cutoffs), 3), dtype=np.int64)
    for number in numbers:
        pre = read_rgb(DATA / "pre" / f"{number:02d}.jpg")
        truth = np.asarray([box for box, _ in load_regions(number)], dtype=np.float32).reshape(-1, 4)
        groups = [localization(model.detector, *detection_image(pre, scale), cutoffs[0], model.device)
                  for scale in SCALES]
        for i, option in enumerate(options):
            boxes, scores = groups[option[0]] if len(option) == 1 else merge_detections([groups[k] for k in option])
            for j, threshold in enumerate(cutoffs):
                selected = boxes[scores >= threshold]
                matched, _ = match(selected, truth)
                totals[i, j] += len(matched), len(selected), len(truth)
    candidates = []
    for i, option in enumerate(options):
        for j, threshold in enumerate(cutoffs):
            matched, predicted, annotated = map(int, totals[i, j])
            candidates.append({"threshold": threshold, "scales": [SCALES[k] for k in option],
                               "box_f1": 2 * matched / max(1, predicted + annotated),
                               "matched": matched, "predicted": predicted, "annotated": annotated})
    best = max(candidates, key=lambda row: (row["box_f1"], -len(row["scales"])))
    print(f"Turkey localization validation: {json.dumps(best)}", flush=True)
    return best, candidates


def collect(model, numbers, split):
    all_scores, targets = [], []
    counts = np.zeros(3, dtype=np.int64)
    missed = np.zeros(3, dtype=np.int64)
    for i, number in enumerate(numbers, 1):
        before, after = paired(number)
        boxes, scores = model.automatic(before, after, *scene(number))
        annotated = load_regions(number)
        truth = np.asarray([box for box, _ in annotated], dtype=np.float32).reshape(-1, 4)
        grade = np.asarray([label for _, label in annotated], dtype=np.int8)
        matched, assigned = match(boxes, truth)
        known = np.full(len(boxes), -1, dtype=np.int8)
        known[matched] = grade[assigned]
        unmatched = np.ones(len(grade), dtype=bool)
        unmatched[assigned] = False
        missed += np.bincount(grade[unmatched], minlength=3)
        counts += len(matched), len(boxes), len(truth)
        all_scores.append(scores)
        targets.append(known)
        print(f"Turkey automatic {split}: {i}/{len(numbers)}, {len(matched)}/{len(truth)} matched", flush=True)
    return np.concatenate(all_scores), np.concatenate(targets), missed, counts


def rename_classes(results):
    results["classes"] = dict(zip(CLASSES, results["classes"].values()))


def evaluate(model, splits, out, split):
    if split == "val":
        model.loc, candidates = choose_localization(model, splits[split])
    scores, target, missed, counts = collect(model, splits[split], split)
    if split == "val":
        quantiles = sorted({*QUANTILES, *np.arange(.06, .95, .03)})
        config = tune_automatic_thresholds(scores, target, missed, quantiles=quantiles)
        rename_classes(config["validation"])
        config.update(localization=model.loc, localization_candidates=candidates)
        destination = out / "automatic_thresholds.json"
    else:
        cfg = json.loads((out / "automatic_thresholds.json").read_bytes())
        graded = predict_grade(scores, np.asarray(cfg["thresholds"]))
        known = target >= 0
        table = np.bincount(3 * target[known].astype(np.int32) + graded[known], minlength=9).reshape(3, 3)
        classify = grade_metrics(table, missed, np.bincount(graded[~known], minlength=3))
        rename_classes(classify)
        config = {"thresholds": cfg["thresholds"], "classification": classify, "localization": model.loc}
        destination = out / "test.json"
    matches, proposals, annotated = map(int, counts)
    config.update({"box_f1": 2 * matches / max(1, proposals + annotated), "matched": matches,
                   "predicted": proposals, "annotated": annotated, "scenes": len(splits[split])})
    write_json(destination, config)
    print(json.dumps(config, indent=2), flush=True)


def predict_pair(model, args):
    before, after = read_rgb(args.pre), read_rgb(args.post)
    boxes, scores = model.automatic(before, after, *detection_image(before))
    threshold = json.loads((args.out / "automatic_thresholds.json").read_bytes())["thresholds"]
    grades = predict_grade(scores, np.asarray(threshold))
    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    overlay = Image.fromarray(after)
    draw = ImageDraw.Draw(overlay)
    with (args.prediction_dir / "grades.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["left", "top", "right", "bottom", "grade", "score_A", "score_B", "score_C"])
        for box, grade, values in zip(boxes, grades, scores):
            draw.rectangle(box.tolist(), outline=COLORS[grade], width=3)
            writer.writerow([*box.tolist(), CLASSES[grade], *values.tolist()])
    overlay.save(args.prediction_dir / "grades_overlay.png")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("train", "calibrate", "test", "image"))
    cli.add_argument("--out", type=Path, default=OUT)
    cli.add_argument("--init", type=Path, default=ROOT / "run/swin_loc_mix/detector.pt")
    cli.add_argument("--epochs", type=int, default=20)
    cli.add_argument("--steps", type=int, default=122)
    cli.add_argument("--batch", type=int, default=4)
    cli.add_argument("--lr", type=float, default=.00017)
    cli.add_argument("--seconds", type=int, default=750)
    cli.add_argument("--pre", type=Path)
    cli.add_argument("--post", type=Path)
    cli.add_argument("--prediction-dir", type=Path)
    args = cli.parse_args()
    if min(args.epochs, args.steps, args.batch, args.seconds, args.lr) <= 0:
        cli.error("training parameters must be positive")
    if args.action == "image" and None in (args.pre, args.post, args.prediction_dir):
        cli.error("image requires --pre, --post and --prediction-dir")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads((DATA / "split.json").read_bytes())["splits"]
    if args.action == "train":
        args.aux_buildings = 0.
        train(args, device, [training_records(splits[split]) for split in ("train", "val")],
              image_loader=lambda number: scene(number)[0])
    elif args.action == "image":
        predict_pair(Assessment(args.out, device), args)
    else:
        evaluate(Assessment(args.out, device), splits, args.out, "val" if args.action == "calibrate" else "test")
