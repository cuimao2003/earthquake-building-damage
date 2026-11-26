"""Fine-tune DINOv2 directly on the Mexico development imagery."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score
import torch

from classifier import RUN, calibrate, load_data, save_model
from features import SIZE, read_record
from finetune_turkey import Patches, load, predict, train
from inventory import ROOT, write_json
from select_mexico import head_scores, likelihood_pair


STORE = ROOT / "data/mexico_patches"
OUT = RUN / "mexico_direct_finetune"
CHECKPOINT = "mexico_vit.pt"


class AuxiliaryPatches(Patches):
    def __init__(self, archive):
        with np.load(archive, allow_pickle=False) as z:
            self.rgb, self.roi, self.change, self.grade = (z[k] for k in ("rgb", "roi", "texture", "grade"))


def prepare(rows, split):
    STORE.mkdir(exist_ok=True)
    original = load_data(split)
    total = len(original["grade"])
    prefix = STORE / split
    images = np.lib.format.open_memmap(prefix.with_name(split + "_rgb.npy"), mode="w+", dtype=np.uint8,
                                      shape=(total, 2, SIZE, SIZE, 3))
    masks = np.lib.format.open_memmap(prefix.with_name(split + "_roi.npy"), mode="w+", dtype=np.uint8,
                                     shape=(total, SIZE, SIZE))
    grades, points, chips = [], [], []
    start = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for offset in range(0, len(rows), 6):
            group = rows[offset:offset + 6]
            results = pool.map(read_record, group, [72 if split == "train" else None] * len(group),
                               ["phase"] * len(group))
            for row, (rgb, roi, change, grade, uid, _) in zip(group, results):
                end = start + len(grade)
                if end > start:
                    if not np.array_equal(original["uid"][start:end], uid):
                        raise ValueError(f"Unexpected tile building order for {row['chip']}")
                    images[start:end] = np.stack(rgb)
                    masks[start:end] = np.stack(roi)
                grades.extend(grade)
                points.extend(change)
                chips.extend([row["chip"]] * len(grade))
                start = end
            if offset % 30 == 0 or offset + 6 >= len(rows):
                print(f"Mexico crop cache {split}: {start}/{total}", flush=True)
    if start != total or not np.array_equal(original["grade"], grades):
        raise ValueError(f"Invalid patch cache for {split}")
    images.flush()
    masks.flush()
    np.savez(prefix.with_suffix(".npz"), change=np.stack(points), grade=np.asarray(grades), chip=np.asarray(chips))


def assess(device, out):
    encoder, head, state = load(out / CHECKPOINT, device)
    data = Patches("val", SIZE, STORE)
    fresh = predict(encoder, head, data, device, batch=64)
    _, minor = head_scores(RUN / "mexico_herat_only")
    _, severe = head_scores(RUN / "mexico_transfer")
    results = []
    for name, a, b in (("previous", minor, severe), ("direct", fresh, fresh),
                       ("direct_minor", fresh, severe), ("direct_severe", minor, fresh)):
        scores = likelihood_pair(a, b)
        thresholds, validation = calibrate(scores, data.grade)
        result = {"candidate": name, "thresholds": thresholds, "validation": validation,
                  "average_precision": [float(average_precision_score(data.grade == k, scores[:, k]))
                                        for k in (1, 2)]}
        results.append(result)
        print(f"{name}: macro F1 {validation['macro_f1']:.4f} and AP {result['average_precision']}", flush=True)
    thresholds, val = calibrate(fresh, data.grade)
    save_model(out / "head.pt", head, epoch=state["epoch"])
    np.save(out / "val_head_scores.npy", fresh)
    write_json(out / "selection.json", {"selected": {"texture_weight": 0., "thresholds": thresholds,
                                                    "validation": val}, "registration": "phase",
                                       "transferred_encoder": True, "encoder_checkpoint": str((out / CHECKPOINT).resolve())})
    write_json(out / "validation_comparison.json", results)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--epochs", type=int, default=12)
    cli.add_argument("--steps", type=int, default=78)
    cli.add_argument("--out", type=Path, default=OUT)
    cli.add_argument("--aux-crops", type=Path)
    args = cli.parse_args()
    if min(args.epochs, args.steps) < 1:
        cli.error("epochs and steps must be positive")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    args.out.mkdir(exist_ok=True, parents=True)
    index = json.loads((ROOT / "data/manifest.json").read_bytes())
    for split in ("train", "val"):
        if not (STORE / f"{split}.npz").is_file():
            prepare(index[split], split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.seed, args.patch_size, args.init = 48123, SIZE, None
    args.blocks, args.lr, args.batch = 2, .000032, 32
    args.local_evidence, args.partial_augment = False, False
    auxiliary = AuxiliaryPatches(args.aux_crops) if args.aux_crops else None
    train(args, device, STORE, sample_fractions=(.65, .23), checkpoint_name=CHECKPOINT, auxiliary=auxiliary)
    assess(device, args.out)
