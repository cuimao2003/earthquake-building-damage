"""Evaluate a fixed checkpoint against the held-out earthquake imagery."""

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import affinity, wkt
import torch

from evaluation import classify, confusion, metrics, predict_map
from infer import COLORS, RUN, load_model, overlay, save_prediction
from prepare import CLASSES, DATA, paint, write_json
from region_model import load_region_model, predict_regions


def annotated_buildings(record, predicted):
    # The annotated footprint is used only to associate predicted pixels with
    # a building for this diagnostic, never as an input to inference.
    label_path = Path(record["source"]) / "labels" / (record["chip"] + "_post_disaster.json")
    raw = json.loads(label_path.read_bytes())["features"]["xy"]
    indexed = {entry["properties"]["uid"]: entry for entry in raw}
    matrix = np.zeros((4, 4), np.int64)
    for building in record["buildings"]:
        label = building["class_id"]
        if label == 255:
            continue
        shape = wkt.loads(indexed[building["uid"]]["wkt"])
        x0, y0, x1, y1 = building["bounds"]
        left = max(0, int(np.floor(x0)))
        top = max(0, int(np.floor(y0)))
        right = min(predicted.shape[1], int(np.ceil(x1)) + 1)
        bottom = min(predicted.shape[0], int(np.ceil(y1)) + 1)
        if left >= right or top >= bottom:
            continue
        mask = np.zeros((bottom - top, right - left), np.uint8)
        paint(mask, affinity.translate(shape, -left, -top), 1)
        grades = predicted[top:bottom, left:right][mask > 0]
        if not grades.size:
            continue
        values = np.bincount(grades, minlength=4)
        grade = 0 if 2 * values[0] > grades.size else 1 + np.argmax(values[1:])
        matrix[label, grade] += 1
    return matrix


def comparison(path, chip, prediction, title):
    size = 410
    padding = 24
    gap = 12
    y0 = 88
    page = Image.new("RGB", (2 * size + 2 * padding + gap, 2 * size + y0 + 124), "#101923")
    draw = ImageDraw.Draw(page)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    zh = font_path.exists()
    font = ImageFont.truetype(str(font_path), 19) if zh else ImageFont.load_default()
    title_font = ImageFont.truetype(str(font_path), 24) if zh else font
    draw.text((padding, 18), title, font=title_font, fill="#edf4fa")
    panels = [chip[..., :3], chip[..., 3:6],
              overlay(chip[..., 3:6], np.where(chip[..., 6] == 255, 0, chip[..., 6])),
              overlay(chip[..., 3:6], prediction)]
    captions = ("震前影像", "震后影像", "人工标注", "模型预测") if zh else (
        "Before", "After", "Annotations", "Prediction")
    for i, (pixels, label) in enumerate(zip(panels, captions)):
        x = padding + (size + gap) * (i % 2)
        y = y0 + (size + 41) * (i // 2)
        draw.text((x, y - 30), label, font=font, fill="#d6e5ef")
        page.paste(Image.fromarray(np.ascontiguousarray(pixels)).resize(
            (size, size), Image.Resampling.LANCZOS), (x, y))
    labels = ("未受损", "轻损", "重损或毁坏") if zh else ("No damage", "Minor", "Major or destroyed")
    y = page.height - 51
    for i, label in enumerate(labels, 1):
        x = padding + 220 * (i - 1)
        draw.rounded_rectangle((x, y, x + 21, y + 21), radius=4, fill=tuple(COLORS[i]))
        draw.text((x + 32, y - 3), label, font=font, fill="#e7eef4")
    page.save(path)


def choose_examples(records):
    def severe_count(row):
        count = row["building_counts"]
        return count.get("major-damage", 0) + count.get("destroyed", 0)

    # Choose examples by label diversity and counts alone.
    strongest = max(records, key=lambda r: (severe_count(r), r["building_counts"].get("minor-damage", 0)))
    mild = max([r for r in records if not severe_count(r)] or records,
               key=lambda r: r["building_counts"].get("minor-damage", 0))
    return {strongest["chip"], mild["chip"]}


def evaluate(model, state, cache, records, out, checkpoint, device, batch, regional=None):
    pixel_total = np.zeros((4, 4), np.int64)
    object_total = np.zeros_like(pixel_total)
    rows = []
    amp = device.type == "cuda"
    examples = choose_examples(records)
    start = time.monotonic()
    for index, record in enumerate(records, 1):
        name = record["chip"]
        chip = np.load(cache / "earthquake" / record["file"], mmap_mode="r")
        probs = predict_map(model, chip[..., :3], chip[..., 3:6], device, batch, amp,
                            **state.get("inference", {}))
        if regional is None:
            predicted = classify(probs, state["thresholds"])
        else:
            region, options, _ = regional
            predicted = predict_regions(region, options, chip[..., :3], chip[..., 3:6], probs, device)
        matrix = confusion(chip[..., 6], predicted)
        pixel_total += matrix
        objects = annotated_buildings(record, predicted)
        object_total += objects
        current = metrics(matrix)
        rows.append({"chip": name, "footprint_f1": current["footprint"]["f1"],
                     "minor_f1": current["classes"][CLASSES[2]]["f1"],
                     "severe_f1": current["classes"][CLASSES[3]]["f1"],
                     "minor_objects": int(objects[2].sum()), "severe_objects": int(objects[3].sum()),
                     "minor_objects_correct": int(objects[2, 2]), "severe_objects_correct": int(objects[3, 3])})
        if name in examples:
            sample_dir = out / "examples" / name
            save_prediction(sample_dir, chip[..., 3:6], predicted, checkpoint,
                            regional[2] if regional else None)
            comparison(sample_dir / "comparison.png", chip, predicted, name)
        if index % 10 == 0 or index == len(records):
            print(f"Evaluated {index}/{len(records)} tiles", flush=True)
    baseline = pixel_total.sum(axis=1)
    locate_baseline = np.zeros_like(pixel_total)
    locate_baseline[:, 0] = pixel_total[:, 0]
    locate_baseline[:, 1] = baseline - locate_baseline[:, 0]
    report = {"tiles": len(records), "epoch": state["epoch"], "thresholds": state["thresholds"],
              "inference": state.get("inference", {}),
              "seconds": round(time.monotonic() - start, 1), "pixel": metrics(pixel_total),
              "annotated_buildings": metrics(object_total),
              "same_footprints_all_intact_baseline": metrics(locate_baseline),
              "example_chips": sorted(examples)}
    if regional:
        report["dense_thresholds"] = report["thresholds"]
        report["thresholds"] = regional[1]["thresholds"]
        report["region_epoch"] = regional[1]["epoch"]
        report["region_blend"] = regional[1]["blend"]
        report["region_harmonize"] = regional[1].get("harmonize", False)
        report["region_checkpoint"] = str(regional[2])
    write_json(out / "test_metrics.json", report)
    with (out / "test_by_tile.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["pixel"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DATA)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "damage_model.pt")
    parser.add_argument("--region-checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=RUN)
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_model(args.checkpoint, device)
    records = json.loads((args.cache / "earthquake.json").read_text(encoding="utf-8"))["test"]
    regional = None
    if args.region_checkpoint:
        path = args.region_checkpoint.resolve()
        regional = (*load_region_model(path, device), path)
    evaluate(model, state, args.cache, records, args.out, args.checkpoint.resolve(), device, args.batch,
             regional)


if __name__ == "__main__":
    main()
