"""Evaluate a localized roof-change feature set on Mexico development data."""

import json
from pathlib import Path

import cv2
import numpy as np
from shapely import wkt
from sklearn.metrics import average_precision_score

from classifier import RUN, calibrate, load_data, scene_features, train_texture
from features import ALIGNED, align, patch_pair, read_rgb
from inventory import ROOT, write_json
from select_mexico import head_scores, likelihood_pair


FEATURE_COUNT = 53


def descriptors(pre, post, bounds):
    views, roof = patch_pair(pre, post, bounds, extent_factor=2.6, size=56)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    outside = ~cv2.dilate(roof.astype(np.uint8), kernel).astype(bool)
    gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255 for image in views]
    highpass = [image - cv2.GaussianBlur(image, (0, 0), 2.2) for image in gray]
    inner = np.s_[8:-8, 8:-8]
    reference = highpass[0][inner]
    context = outside[inner].astype(np.uint8)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if context.sum() < 12:
        missing = np.zeros(FEATURE_COUNT, np.float32)
        missing[:4] = np.log1p(width), np.log1p(height), -1., -1.
        return missing
    similarity = cv2.matchTemplate(highpass[1], reference.copy(), cv2.TM_CCORR_NORMED, mask=context)
    similarity = np.nan_to_num(similarity, nan=-1., posinf=-1., neginf=-1.).clip(-1, 1)
    y, x = np.unravel_index(similarity.argmax(), similarity.shape)
    values = [np.log1p(width), np.log1p(height), similarity[8, 8], similarity[y, x],
              np.hypot(x - 8, y - 8) / 8]
    for left, top in ((8, 8), (x, y)):
        after = np.s_[top:top + 40, left:left + 40]
        change = highpass[1][after] - reference
        difference = gray[1][after] - gray[0][inner]
        keep = roof[inner]
        ring = outside[inner]
        for a, b in ((reference, highpass[1][after]), (gray[0][inner], gray[1][after])):
            for mask in (keep, ring):
                av = a[mask] - a[mask].mean()
                bv = b[mask] - b[mask].mean()
                values.append(float(av @ bv / max(.0001, np.linalg.norm(av) * np.linalg.norm(bv))))
        change = np.abs(change)
        difference = np.abs(difference - np.median(difference[ring]))
        for residual in (change, difference):
            inside = np.quantile(residual[keep], [.25, .5, .75, .90, .97])
            surround = np.quantile(residual[ring], [.25, .5, .75, .90, .97])
            values.extend(inside)
            values.extend(inside - surround)
    return np.asarray(values, np.float32)


def scene_values(row, wanted):
    root, chip = Path(row["source"]), row["chip"]
    pre = read_rgb(root / "images" / f"{chip}_pre_disaster.png")
    cache = ALIGNED / f"{chip}.npz"
    if cache.is_file():
        with np.load(cache) as registered:
            post = registered["post"]
    else:
        post = align(pre, read_rgb(root / "images" / f"{chip}_post_disaster.png"), "keypoint")[0]
    labels = json.loads((root / "labels" / f"{chip}_post_disaster.json").read_bytes())
    bounds = {item["properties"]["uid"]: wkt.loads(item["wkt"]).bounds
              for item in labels["features"]["xy"] if item["properties"]["uid"] in wanted}
    return np.stack([descriptors(pre, post, bounds[uid]) for uid in wanted])


def prepare(rows, split, out):
    data = load_data(split)
    values = np.empty((len(data["grade"]), FEATURE_COUNT), np.float32)
    for index, row in enumerate(rows, 1):
        indices = np.flatnonzero(data["chip"] == row["chip"])
        if len(indices):
            values[indices] = scene_values(row, data["uid"][indices].tolist())
        if index % 30 == 0 or index == len(rows):
            print(f"Local roof change {split}: {index}/{len(rows)}", flush=True)
    combined = np.concatenate((data["change"], scene_features(values, data["chip"])), axis=1)
    np.save(out / f"{split}_change.npy", combined)
    return {"grade": data["grade"], "change": combined}


def main():
    cv2.setNumThreads(1)
    out = RUN / "mexico_local_change_audit"
    out.mkdir(exist_ok=True)
    index = json.loads((ROOT / "data/manifest.json").read_bytes())
    train, val = [prepare(index[split], split, out) for split in ("train", "val")]
    truth = val["grade"]
    local = train_texture(train, val, out)
    _, light = head_scores(RUN / "mexico_herat_only")
    _, severe = head_scores(RUN / "mexico_transfer")
    results = []
    for a, b in ((0, 0), (.2, 0), (.5, 0), (1, 0), (0, .2), (0, .5), (0, 1), (.35, .35), (1, 1)):
        scores = likelihood_pair(light * (1 - a) + local * a, severe * (1 - b) + local * b)
        limits, result = calibrate(scores, truth)
        results.append({"local_texture_weight": [a, b], "thresholds": limits,
                        "validation": result, "average_precision": [float(average_precision_score(
                            truth == grade, scores[:, grade])) for grade in (1, 2)]})
        print(f"Texture weights {a:.2f}/{b:.2f}: validation macro F1 {result['macro_f1']:.4f}", flush=True)
    write_json(out / "validation.json", results)


if __name__ == "__main__":
    main()
