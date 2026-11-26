"""Validate multi-view Swin building boxes on Mexico development imagery."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from detector import match
from features import read_rgb
from inventory import ROOT, write_json
from swin_locator import TILE, load_swin, swin_boxes


OUT = ROOT / "run/mexico_location_fusion"
NETWORKS = {"base": "swin_loc_mix", "small": "swin_loc_small"}
LOOSE = .14


@lru_cache(maxsize=2)
def load_network(network, device):
    return load_swin(ROOT / "run" / NETWORKS[network] / "detector.pt", device)


def proposals(image, view, device):
    network, mirrored = view
    model, state = load_network(network, device)
    boxes, scores = swin_boxes(model, image[:, ::-1] if mirrored else image, device, LOOSE,
                              tile=state.get("tile_size", TILE))
    if mirrored:
        boxes[:, [0, 2]] = image.shape[1] - boxes[:, [2, 0]]
    return boxes, scores


def consensus(left, right, association=.52, penalty=.72):
    boxes, scores = left
    extra, confidence = right
    i, j = match(boxes, extra, association)
    unmatched = np.ones(len(boxes), dtype=bool)
    unmatched[i] = False
    other = np.ones(len(extra), dtype=bool)
    other[j] = False
    weighted = scores[i] + confidence[j]
    pooled = (boxes[i] * scores[i, None] + extra[j] * confidence[j, None]) / weighted[:, None]
    combined = np.concatenate((pooled, boxes[unmatched], extra[other]))
    certainty = np.concatenate((weighted * .5, scores[unmatched] * penalty, confidence[other] * penalty))
    keep = nms(torch.from_numpy(combined), torch.from_numpy(certainty), .47).numpy()
    return combined[keep], certainty[keep]


def fused_boxes(image, settings, device):
    parts = [proposals(image, tuple(v), device) for v in settings["views"]]
    if len(parts) == 1:
        boxes, score = parts[0]
    else:
        boxes, score = consensus(*parts, association=settings["association"], penalty=settings["penalty"])
    return boxes[score >= settings["threshold"]]


def choices(views):
    for view in views:
        yield {"views": [view]}
    for a, b in ((views[0], views[1]), (views[0], views[2]), (views[2], views[3])):
        for association in (.48, .57):
            for penalty in (.62, .80, .96):
                yield {"views": [a, b], "association": association, "penalty": penalty}


def collect(records, views, device):
    from infer_retrain import ground_truth

    data = []
    for i, row in enumerate(records, 1):
        source = Path(row["source"]) / "images" / f"{row['chip']}_pre_disaster.png"
        image = read_rgb(source)
        truth, grade = ground_truth(row)
        candidates = {view: proposals(image, view, device) for view in views}
        data.append((truth, grade, candidates))
        if i % 5 == 0 or i == len(records):
            print(f"Mexico candidate localization {i}/{len(records)}", flush=True)
    return data


def choose(data, options):
    result = []
    cutoffs = (.20, .24, .28, .32, .36, .40, .44, .48, .54, .60, .66)
    for option in options:
        totals = np.zeros((len(cutoffs), 3), dtype=np.int64)
        landed = np.zeros((len(cutoffs), 3), dtype=np.int64)
        for truth, grade, views in data:
            groups = [views[v] for v in option["views"]]
            if len(groups) == 1:
                boxes, scores = groups[0]
            else:
                boxes, scores = consensus(*groups, association=option["association"], penalty=option["penalty"])
            for i, threshold in enumerate(cutoffs):
                selected = boxes[scores >= threshold]
                matched, expected = match(selected, truth)
                totals[i] += len(matched), len(selected), len(truth)
                landed[i] += np.bincount(grade[expected][grade[expected] >= 0], minlength=3)
        for i, threshold in enumerate(cutoffs):
            hits, predicted, annotated = map(int, totals[i])
            result.append({**option, "threshold": threshold,
                           "box_f1": 2 * hits / max(1, predicted + annotated),
                           "matched_by_class": landed[i].tolist(), "matches": hits,
                           "predicted": predicted, "annotated": annotated})
    return sorted(result, key=lambda r: (r["box_f1"], -len(r["views"])), reverse=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=OUT)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((ROOT / "data/manifest.json").read_bytes())
    views = [(name, mirror) for name in NETWORKS for mirror in (False, True)]
    records = collect(manifest["val"], views, device)
    candidates = choose(records, choices(views))
    selected = candidates[0]
    print(json.dumps(selected, indent=2), flush=True)
    write_json(args.out / "validation.json", {"selected": selected, "candidates": candidates})
