"""Evaluate whether direction-aware shadow features help on Mexico development scenes."""

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score

from classifier import RUN, calibrate, load_data, odds, scene_features, train_texture
from finetune_mexico import STORE
from finetune_turkey import Patches
from inventory import ROOT, write_json
from select_mexico import head_scores, likelihood_pair


def sun_angles(row):
    root = Path(row["source"]) / "labels"
    values = []
    for when in ("pre", "post"):
        metadata = json.loads((root / f"{row['chip']}_{when}_disaster.json").read_bytes())["metadata"]
        values.append((float(metadata["sun_azimuth"]), float(metadata["sun_elevation"])))
    return values


def directional_shadow(rgb, roi, angles):
    y, x = np.ogrid[:roi.shape[0], :roi.shape[1]]
    dx = x - roi.shape[1] / 2
    dy = y - roi.shape[0] / 2
    occupied_y, occupied_x = np.nonzero(roi)
    rx = max(1., np.ptp(occupied_x) / 2)
    ry = max(1., np.ptp(occupied_y) / 2)
    radius = np.hypot(dx / rx, dy / ry)
    dist = np.hypot(dx, dy).clip(1)
    descriptors = []
    for image, (azimuth, elevation) in zip(rgb, angles):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
        angle = np.deg2rad(azimuth)
        along = (-np.sin(angle) * dx + np.cos(angle) * dy) / dist
        surround = (radius >= 1.04) & (radius < 1.85)
        reference = gray[surround]
        lo, median, hi = np.quantile(reference, [.15, .5, .85]) if reference.size else (0., 0., 1.)
        contrast = (median - gray) / max(.03, hi - lo)
        values = []
        height_scale = np.tan(np.deg2rad(np.clip(elevation, 10., 80.)))
        for near, far in ((1.04, 1.43), (1.43, 1.85)):
            band = (radius >= near) & (radius < far)
            dark = gray < lo
            for mask in (band & (along > .70), band & (along < -.70), band & (np.abs(along) <= .45)):
                dark_in_mask = mask & dark
                if mask.any():
                    values.extend((float(contrast[mask].mean()), float(dark[mask].mean()),
                                   float(((radius - 1.) * dark_in_mask).sum() / mask.sum() * height_scale)))
                else:
                    values.extend((0., 0., 0.))
        descriptors.append(np.asarray(values, np.float32))
    return np.concatenate((descriptors[0], descriptors[1], descriptors[1] - descriptors[0]))


def prepare(rows, split):
    cache = Patches(split, store=STORE)
    data = load_data(split)
    if not np.array_equal(cache.chip, data["chip"]) or not np.array_equal(cache.grade, data["grade"]):
        raise ValueError("Unexpected cache order")
    metadata = {row["chip"]: sun_angles(row) for row in rows}
    novel = np.stack([directional_shadow(image, roof, metadata[chip])
                      for image, roof, chip in zip(cache.rgb, cache.roi, cache.chip)])
    combined = np.concatenate((data["change"], scene_features(novel, data["chip"])), axis=1)
    print(f"Solar geometry {split}: {len(novel)} buildings", flush=True)
    return {"grade": data["grade"], "change": combined}


def main():
    cv2.setNumThreads(1)
    out = RUN / "mexico_solar_geometry"
    out.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "data/manifest.json").read_bytes())
    train, val = [prepare(manifest[split], split) for split in ("train", "val")]
    _, light = head_scores(RUN / "mexico_herat_only")
    _, severe = head_scores(RUN / "mexico_transfer")
    solar = train_texture(train, val, out)
    truth = val["grade"]
    results = []
    for a, b in ((0, 0), (1, 1), (1, 0), (0, 1), (.35, .35), (.35, 0), (0, .35), (0, .65), (.65, 0)):
        scores = likelihood_pair(light * (1 - a) + solar * a, severe * (1 - b) + solar * b)
        thresholds, validation = calibrate(scores, truth)
        avg_precision = [float(average_precision_score(truth == grade, odds(scores)[:, grade - 1]))
                         for grade in (1, 2)]
        results.append({"solar_weight": [a, b], "thresholds": thresholds, "validation": validation,
                        "average_precision": avg_precision})
        print(f"Solar weights {a:.2f}/{b:.2f}: macro F1 {validation['macro_f1']:.4f}", flush=True)
    write_json(out / "validation.json", results)


if __name__ == "__main__":
    main()
