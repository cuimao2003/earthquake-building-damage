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
from features import SIZE, WEIGHTS, read_record
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


def prepare(rows, split, store, size):
    store.mkdir(exist_ok=True)
    original = load_data(split)
    total = len(original["grade"])
    prefix = store / split
    images = np.lib.format.open_memmap(prefix.with_name(split + "_rgb.npy"), mode="w+", dtype=np.uint8,
                                      shape=(total, 2, size, size, 3))
    masks = np.lib.format.open_memmap(prefix.with_name(split + "_roi.npy"), mode="w+", dtype=np.uint8,
                                     shape=(total, size, size))
    grades, points, chips = [], [], []
    start = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for offset in range(0, len(rows), 6):
            group = rows[offset:offset + 6]
            results = pool.map(read_record, group, [72 if split == "train" else None] * len(group),
                               ["phase"] * len(group), [False] * len(group), [size] * len(group))
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


def assess(device, out, store):
    encoder, head, state = load(out / CHECKPOINT, device)
    size = state["patch_size"]
    data = Patches("val", size, store)
    local_evidence = state.get("local_evidence", False)
    batch = 32 if size > 112 or encoder.width > 768 else 64
    fresh = predict(encoder, head, data, device, batch=batch, local_evidence=local_evidence)
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
                                       "patch_size": size, "transferred_encoder": True,
                                       "local_evidence": local_evidence,
                                       "encoder_checkpoint": str((out / CHECKPOINT).resolve())})
    write_json(out / "validation_comparison.json", results)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--epochs", type=int, default=12)
    cli.add_argument("--steps", type=int, default=78)
    cli.add_argument("--out", type=Path, default=OUT)
    cli.add_argument("--aux-crops", type=Path)
    cli.add_argument("--weights", type=Path, default=WEIGHTS)
    cli.add_argument("--patch-size", type=int, choices=(112, 168), default=SIZE)
    cli.add_argument("--blocks", type=int, choices=(1, 2, 3, 4), default=2)
    cli.add_argument("--batch", type=int, default=32)
    cli.add_argument("--lr", type=float, default=.000032)
    cli.add_argument("--local-evidence", action="store_true", help="Emphasize anomalous roof tokens")
    args = cli.parse_args()
    if min(args.epochs, args.steps, args.lr) <= 0 or args.batch < 10:
        cli.error("epochs, steps and learning rate must be positive and batch must be at least ten")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    args.out.mkdir(exist_ok=True, parents=True)
    store = STORE if args.patch_size == SIZE else STORE.with_name(f"{STORE.name}_{args.patch_size}")
    index = json.loads((ROOT / "data/manifest.json").read_bytes())
    for split in ("train", "val"):
        if not (store / f"{split}.npz").is_file():
            prepare(index[split], split, store, args.patch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.seed, args.init = 48123, None
    args.partial_augment = False
    auxiliary = AuxiliaryPatches(args.aux_crops) if args.aux_crops else None
    train(args, device, store, sample_fractions=(.65, .23), checkpoint_name=CHECKPOINT, auxiliary=auxiliary)
    assess(device, args.out, store)
