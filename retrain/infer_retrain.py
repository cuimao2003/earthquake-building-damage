"""Run paired building assessment and evaluate the locked Mexico holdout."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw
from shapely import wkt
import torch

from classifier import QUANTILES, load_head, odds, predict_grade, probabilities, report, scene_features
from detector import detect, load_detector, match
from dino import load_dino
from evaluation import confusion, metrics, predict_map
from features import WEIGHTS, align, embed, patch_pair, read_rgb, signal
from infer import load_model
from inventory import LABELS, ROOT, write_json
from prepare import paint
from region_model import (dense_region_scores, extract_proposals, load_region_model,
                          proposal_crops, region_scores, score_crops)
from select_mexico import likelihood_pair

COLORS = ((82, 179, 232), (255, 194, 49), (255, 72, 85))


def ground_truth(row):
    labels = Path(row["source"]) / "labels" / f"{row['chip']}_post_disaster.json"
    raw = json.loads(labels.read_bytes())["features"]["xy"]
    return (np.array([wkt.loads(b["wkt"]).bounds for b in raw], np.float32).reshape(-1, 4),
            np.asarray([LABELS.get(b["properties"].get("subtype"), -1) for b in raw], np.int8))


def paired_images(row):
    root = Path(row["source"]) / "images"
    return [read_rgb(root / f"{row['chip']}_{date}_disaster.png") for date in ("pre", "post")]


def load_localizers(device):
    center, c = load_detector(ROOT / "run/centers/detector.pt", device)
    footprint, f = load_model(ROOT.parent / "run/damage_model.pt", device)
    return center, c, footprint, f


@lru_cache(maxsize=2)
def load_new_localizer(kind, device):
    from swin_locator import load_swin
    folder = "swin_loc_mix" if kind == "swin_mix" else "swin_loc"
    return load_swin(ROOT / "run" / folder / "detector.pt", device)


def localize(kind, positions, pre, post, device):
    center, c, footprint, f = positions
    if kind == "swin_consensus":
        from mexico_location_fusion import OUT, fused_boxes
        settings = json.loads((OUT / "validation.json").read_bytes())["selected"]
        return fused_boxes(pre, settings, device)
    if kind == "center":
        boxes, _ = detect(center, pre, device, c["threshold"])
        return boxes
    if kind in ("swin", "swin_mix"):
        from swin_locator import swin_boxes
        model, state = load_new_localizer(kind, device)
        boxes, _ = swin_boxes(model, pre, device, state["threshold"])
        return boxes
    foot = predict_map(footprint, pre, post, device, batch=8, amp=device.type == "cuda",
                       **f.get("inference", {}))[0]
    _, proposals = extract_proposals(foot)
    return proposals.astype(np.float32)


def choose_localizer(records, localizers, device):
    candidates = {}
    options = ["center", "footprint"]
    if (ROOT / "run/swin_loc/detector.pt").is_file():
        options += ["swin"]
    if (ROOT / "run/swin_loc_mix/detector.pt").is_file():
        options += ["swin_mix"]
    for kind in options:
        totals = np.zeros(3, np.int64)
        for row in records:
            pre, post = paired_images(row)
            boxes = localize(kind, localizers, pre, post, device)
            exact, _ = ground_truth(row)
            predicted, _ = match(boxes, exact)
            totals += [len(predicted), len(boxes), len(exact)]
        p, n_predict, n_true = map(int, totals)
        candidates[kind] = {"matches": p, "predicted": n_predict, "annotated": n_true,
                            "box_f1_iou_50": 2 * p / max(1, n_predict + n_true)}
        print(f"Mexico locator {kind}: F1 {candidates[kind]['box_f1_iou_50']:.4f}", flush=True)
    best = max(candidates, key=lambda k: candidates[k]["box_f1_iou_50"])
    result = {"selected": best, "validation": candidates}
    write_json(ROOT / "run/localizer_selection.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return best


class Grader:
    def __init__(self, path, device, encoder_weights=None):
        self.device = device
        self.encoder = load_dino(WEIGHTS, device)
        if encoder_weights:
            checkpoint = torch.load(encoder_weights, map_location="cpu")
            self.encoder.load_state_dict(checkpoint["encoder"])
        self.head, _ = load_head(path / "head.pt", device)
        config = json.loads((path / "selection.json").read_bytes())
        if bool(encoder_weights) != config.get("transferred_encoder", False):
            raise ValueError("Pass the fine-tuned encoder only with a transferred classifier")
        self.selection = config["selected"]
        self.context = config.get("scene_context", False)
        self.registration = config.get("registration", "phase")
        self.texture = (joblib.load(path / "texture.joblib")
                        if self.selection["texture_weight"] else None)
        self.aligned = None

    def classify(self, pre, post, boxes):
        if not len(boxes):
            return np.zeros(0, np.int64), np.zeros((0, 3), np.float32)
        if self.aligned is None or self.aligned[0] is not pre or self.aligned[1] is not post:
            self.aligned = (pre, post, align(pre, post, self.registration)[0])
        registered = self.aligned[2]
        visual, signals = [], []

        def prepare(bounds):
            im, mask = patch_pair(pre, registered, bounds)
            return im, mask, signal(im, mask)

        with ThreadPoolExecutor(max_workers=3) as workers:
            for i in range(0, len(boxes), 72):
                crops, roi, stats = zip(*workers.map(prepare, boxes[i:i + 72]))
                a, cross = embed(self.encoder, np.stack(crops), np.stack(roi), self.device)
                x = np.concatenate((np.stack(stats), cross), axis=1)
                visual.append(a)
                signals.append(x)
        visual, values = np.concatenate(visual), np.concatenate(signals)
        if self.context:
            values = scene_features(values)
        scores = probabilities(self.head, visual, values, self.device)
        if self.texture is not None:
            weight = self.selection["texture_weight"]
            scores = (1 - weight) * scores + weight * self.texture.predict_proba(values)
        return predict_grade(scores, np.asarray(self.selection["thresholds"])), scores


class SelectedGrader:
    def __init__(self, path, device):
        self.device = device
        config = json.loads((path / "selection.json").read_bytes())["selected"]
        self.thresholds = np.asarray(config["thresholds"])
        self.calibrated_localizer = None
        if (path / "automatic_thresholds.json").exists():
            calibrated = json.loads((path / "automatic_thresholds.json").read_bytes())
            self.automatic_thresholds = np.asarray(calibrated["thresholds"])
            self.calibrated_localizer = calibrated.get("localizer")
        else:
            self.automatic_thresholds = self.thresholds
        self.light = Grader(Path(config["minor"]["model"]), device, config["minor"]["encoder"])
        self.severe = (self.light if config["major"] == config["minor"] else
                       Grader(Path(config["major"]["model"]), device, config["major"]["encoder"]))

    def classify(self, pre, post, boxes):
        _, minor = self.light.classify(pre, post, boxes)
        _, major = self.severe.classify(pre, post, boxes) if self.severe is not self.light else (None, minor)
        if len(minor) == 0:
            return np.zeros(0, np.int64), minor
        probs = likelihood_pair(minor, major)
        return predict_grade(probs, self.automatic_thresholds), probs


class PreviousGrader:
    def __init__(self, device):
        self.device = device
        old = ROOT.parent / "run"
        self.dense, self.settings = load_model(old / "damage_model.pt", device)
        self.region, self.state = load_region_model(old / "region_final/region_model.pt", device)

    def locate(self, before, after):
        self.probs = predict_map(self.dense, before, after, self.device, batch=8,
                                 amp=self.device.type == "cuda", **self.settings.get("inference", {}))
        self.regions, boxes = extract_proposals(self.probs[0])
        return boxes

    def classify(self, before, after, boxes):
        inputs = proposal_crops(before, after, boxes)
        scores = score_crops(self.region, inputs, self.device, harmonize=self.state.get("harmonize", False))
        dense = dense_region_scores(self.probs, self.regions, len(boxes))
        scores = region_scores(scores, dense, self.state["blend"])[1:]
        minor, severe = scores[:, 1:].T
        threshold = self.state["thresholds"]
        pred = np.where((minor >= severe) & (minor >= threshold["minor"]), 1,
                        np.where((severe > minor) & (severe >= threshold["severe"]), 2, 0))
        return pred, scores


def comparison(pre, post, annotated, gt, boxes, labels, destination):
    annotated = annotated[gt >= 0]
    gt = gt[gt >= 0]
    width = 422
    nrows = [Image.fromarray(im).resize((width, width)) for im in (pre, post, post, post)]
    for panel, bxs, grades in ((nrows[2], annotated, gt), (nrows[3], boxes, labels)):
        draw = ImageDraw.Draw(panel)
        factor = np.array([width / pre.shape[1], width / pre.shape[0]] * 2)
        for box, level in zip(bxs, grades):
            draw.rectangle(tuple((box * factor).tolist()), outline=COLORS[level], width=2)
    page = Image.new("RGB", (2 * width + 30, 2 * width + 90), "#13202b")
    painter = ImageDraw.Draw(page)
    for i, (panel, text) in enumerate(zip(nrows, ("Pre-event", "Post-event", "Human labels", "Prediction"))):
        x, y = 10 + (width + 10) * (i % 2), 26 + (width + 30) * (i // 2)
        page.paste(panel, (x, y))
        painter.text((x, y - 17), text, fill="#f2f6f9")
    for i, caption in enumerate(("No visible damage", "Minor", "Major or destroyed")):
        y = page.height - 25
        painter.rectangle((10 + i * 240, y, 23 + i * 240, y + 13), fill=COLORS[i])
        painter.text((30 + i * 240, y + 1), caption, fill="white")
    page.save(destination)


def grade_metrics(confusion, missed, extra):
    result = report(confusion, missed)
    pred = confusion.sum(0) + extra
    truth = confusion.sum(1) + missed
    correct = confusion.diagonal()
    f1 = 2 * correct / np.maximum(1, pred + truth)
    result["macro_f1"] = float(f1.mean())
    result["damaged_macro_f1"] = float(f1[1:].mean())
    result["unmatched_predictions"] = extra.tolist()
    for i, entry in enumerate(result["classes"].values()):
        entry.update({"precision": float(correct[i] / max(1, pred[i])),
                      "f1": float(f1[i]), "predicted": int(pred[i])})
    return result


def rasterize_boxes(shape, boxes, grades):
    raster = np.zeros(shape, np.uint8)
    h, w = shape
    safe = np.rint(boxes).astype(int).clip(0, [w, h, w, h])
    for (x0, y0, x1, y1), grade in zip(safe[::-1], grades[::-1]):
        raster[y0:y1, x0:x1] = int(grade) + 1
    return raster


def paint_labels(shape, record):
    label = Path(record["source"]) / "labels" / f"{record['chip']}_post_disaster.json"
    raw = json.loads(label.read_bytes())["features"]["xy"]
    raster = np.zeros(shape, np.uint8)
    polygons = [(LABELS.get(row["properties"].get("subtype"), 254) + 1, row["wkt"]) for row in raw]
    for grade, geometry in sorted(polygons, key=lambda item: item[0]):
        paint(raster, wkt.loads(geometry), grade)
    return raster


def calibrate_boxes(records, grader, mode, loc, out):
    values, targets = [], []
    missed = np.zeros(3, np.int64)
    for i, row in enumerate(records, 1):
        before, after = paired_images(row)
        proposals = localize(mode, loc, before, after, grader.device)
        truth, labels = ground_truth(row)
        _, scores = grader.classify(before, after, proposals)
        p, t = match(proposals, truth)
        assigned = np.full(len(proposals), -1, np.int8)
        assigned[p] = np.where(labels[t] >= 0, labels[t], -2)
        include = assigned >= -1
        values.append(scores[include])
        targets.append(assigned[include])
        unseen = np.ones(len(truth), bool)
        unseen[t] = False
        missed += np.bincount(labels[unseen & (labels >= 0)], minlength=3)
        if i % 10 == 0 or i == len(records):
            print(f"Threshold validation on automatic boxes: {i}/{len(records)}", flush=True)
    best = tune_automatic_thresholds(np.concatenate(values), np.concatenate(targets), missed)
    best["localizer"] = mode
    write_json(out / "automatic_thresholds.json", best)
    print(json.dumps(best, indent=2), flush=True)


def tune_automatic_thresholds(scores, target, missed, quantiles=QUANTILES):
    margins = odds(scores)
    settings = [np.unique(np.r_[-20., np.quantile(margins[:, i], quantiles), 20.]) for i in range(2)]
    known = target >= 0
    actual = target[known].astype(np.int32)
    best = None
    for light in settings[0]:
        for severe in settings[1]:
            predicted = predict_grade(scores, np.array([light, severe]))
            table = np.bincount(3 * actual + predicted[known], minlength=9).reshape(3, 3)
            extras = np.bincount(predicted[~known], minlength=3)
            result = grade_metrics(table, missed, extras)
            if best is None or result["macro_f1"] > best["validation"]["macro_f1"]:
                best = {"thresholds": [float(light), float(severe)], "validation": result}
    return best


def evaluate(records, grader, mode, loc, out, known_boxes=False, pixel_audit=False):
    totals = np.zeros((3, 3), np.int64)
    unmatched_truth = np.zeros(3, np.int64)
    unmatched_predictions = np.zeros(3, np.int64)
    localization = [0, 0, 0]
    known_confusion = np.zeros((3, 3), np.int64)
    pixels = np.zeros((4, 4), np.int64)
    detail = []
    example = max(records, key=lambda r: r["counts"].get("minor-damage", 0) + r["counts"].get("major-damage", 0))
    for i, row in enumerate(records, 1):
        pre, post = paired_images(row)
        gt, grades = ground_truth(row)
        boxes = (grader.locate(pre, post) if mode == "previous_regions"
                 else localize(mode, loc, pre, post, grader.device))
        estimates, _ = grader.classify(pre, post, boxes)
        if pixel_audit:
            shape = pre.shape[:2]
            pixels += confusion(paint_labels(shape, row), rasterize_boxes(shape, boxes, estimates))
        if known_boxes:
            known = grades >= 0
            oracle, oracle_scores = grader.classify(pre, post, gt[known])
            if isinstance(grader, SelectedGrader):
                oracle = predict_grade(oracle_scores, grader.thresholds)
            np.add.at(known_confusion, (grades[known], oracle), 1)
        matched, assigned = match(boxes, gt)
        localization[0] += len(matched)
        localization[1] += len(boxes)
        localization[2] += len(gt)
        assigned_mask = np.ones(len(gt), bool)
        predicted_mask = np.ones(len(boxes), bool)
        assigned_mask[assigned] = False
        predicted_mask[matched] = False
        keep = grades[assigned] >= 0
        np.add.at(totals, (grades[assigned[keep]], estimates[matched[keep]]), 1)
        unmatched_truth += np.bincount(grades[assigned_mask & (grades >= 0)], minlength=3)
        unmatched_predictions += np.bincount(estimates[predicted_mask], minlength=3)
        detail.append({"chip": row["chip"], "matched": len(matched), "predicted": len(boxes),
                       "annotated": len(gt), "grade_correct": int((grades[assigned[keep]] ==
                                                                          estimates[matched[keep]]).sum())})
        if example["chip"] == row["chip"]:
            background = align(pre, post, "keypoint")[0] if mode != "previous_regions" else post
            comparison(pre, background, gt, grades, boxes, estimates, out / "holdout_comparison.png")
        if i % 10 == 0 or i == len(records):
            print(f"Mexico inference {i}/{len(records)}", flush=True)
    p, n, t = localization
    result = {"scenes": len(records), "localizer": mode, "detection_iou_threshold": .5,
              "box_f1": 2 * p / max(1, n + t),
              "detection_precision": p / max(1, n), "detection_recall": p / max(1, t),
              "detection_counts": localization, "classification": grade_metrics(totals, unmatched_truth,
                                                                                 unmatched_predictions),
              "example": example["chip"]}
    if known_boxes:
        result["given_human_boxes_diagnostic"] = report(known_confusion)
    if pixel_audit:
        result["box_raster_metrics"] = metrics(pixels)
    write_json(out / "end_to_end.json", result)
    with (out / "per_tile.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    print(json.dumps(result, indent=2), flush=True)


def predict_pair(pre, post, grader, mode, localizers, out):
    if pre.shape != post.shape:
        raise ValueError("Pre- and post-earthquake images must have identical dimensions")
    boxes = localize(mode, localizers, pre, post, grader.device)
    labels, scores = grader.classify(pre, post, boxes)
    mask = rasterize_boxes(pre.shape[:2], boxes, labels)
    background, registration = align(pre, post, "keypoint")
    overlay = Image.fromarray(background)
    draw = ImageDraw.Draw(overlay)
    rows = []
    for bounds, grade, score in zip(boxes, labels, scores):
        x0, y0, x1, y1 = np.rint(bounds).astype(int)
        draw.rectangle((x0, y0, x1, y1), outline=COLORS[grade], width=2)
        rows.append([x0, y0, x1, y1, int(grade + 1), *score.tolist()])
    overlay.save(out / "prediction_overlay.png")
    Image.fromarray(mask).save(out / "prediction_mask.png")
    with (out / "buildings.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["left", "top", "right", "bottom", "class_id", "no_damage_score",
                         "minor_score", "major_or_destroyed_score"])
        writer.writerows(rows)
    write_json(out / "prediction.json", {"localizer": mode, "predicted_buildings": len(boxes),
                                          "by_class": np.bincount(labels, minlength=3).tolist(),
                                          "coordinate_frame": "pre_event",
                                          "display_registration": registration,
                                          "shape": list(mask.shape)})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("localizers", "calibrate-boxes", "test", "image", "baseline"))
    cli.add_argument("--classifier", type=Path, default=ROOT / "run/mexico_final")
    cli.add_argument("--encoder", type=Path)
    cli.add_argument("--localizer", choices=("center", "footprint", "swin", "swin_mix", "swin_consensus"))
    cli.add_argument("--out", type=Path, default=ROOT / "run/mexico_holdout")
    cli.add_argument("--pre", type=Path)
    cli.add_argument("--post", type=Path)
    cli.add_argument("--device", choices=("auto", "cpu"), default="auto")
    cli.add_argument("--with-box-diagnostic", action="store_true")
    cli.add_argument("--pixel-audit", action="store_true")
    args = cli.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    manifest = json.loads((ROOT / "data/manifest.json").read_bytes())
    if args.action == "baseline":
        args.out.mkdir(parents=True, exist_ok=True)
        evaluate(manifest["test"], PreviousGrader(device), "previous_regions", None, args.out,
                 pixel_audit=args.pixel_audit)
        raise SystemExit(0)
    localizers = load_localizers(device)
    if args.action == "localizers":
        choose_localizer(manifest["val"], localizers, device)
    else:
        cfg = json.loads((ROOT / "run/localizer_selection.json").read_bytes())
        mode = args.localizer or cfg["selected"]
        args.out.mkdir(parents=True, exist_ok=True)
        selected = json.loads((args.classifier / "selection.json").read_bytes())["selected"]
        grader = (SelectedGrader(args.classifier, device) if "major" in selected else
                  Grader(args.classifier, device, args.encoder))
        if (args.action != "calibrate-boxes" and isinstance(grader, SelectedGrader) and
                grader.calibrated_localizer is not None and grader.calibrated_localizer != mode):
            cli.error(f"The saved automatic thresholds are calibrated for {grader.calibrated_localizer}")
        if args.action == "calibrate-boxes":
            if not isinstance(grader, SelectedGrader):
                cli.error("calibrate-boxes requires a locked Mexico selection")
            calibrate_boxes(manifest["val"], grader, mode, localizers, args.classifier)
        elif args.action == "image":
            if args.pre is None or args.post is None:
                cli.error("image requires --pre and --post")
            predict_pair(read_rgb(args.pre), read_rgb(args.post), grader, mode, localizers,
                         args.out)
        else:
            evaluate(manifest["test"], grader, mode, localizers, args.out,
                     args.with_box_diagnostic, args.pixel_audit)
