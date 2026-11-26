"""Fit a small Transformer on paired crops around individual buildings."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from evaluation import Calibration, predict_map
from infer import RUN, load_model
from prepare import DATA, CLASSES, write_json
from region_model import (CHIP_SIZE, RegionTransformer, building_crop, dense_region_scores,
                          extract_proposals, load_region_model, proposal_crops, read_pair,
                          region_scores, score_crops, tensor_inputs)
from select_checkpoint import FINAL_THRESHOLDS
from train import augment_image, save_checkpoint, transform_chip


def prepare_validation(cache, records, model, state, device):
    work = cache / "region_val"
    work.mkdir(exist_ok=True)
    samples = []
    for i, record in enumerate(records, 1):
        chip = np.load(cache / "earthquake" / record["file"], mmap_mode="r")
        before, after = chip[..., :3], chip[..., 3:6]
        probs = predict_map(model, before, after, device, batch=8, amp=device.type == "cuda",
                            **state.get("inference", {}))
        instances, boxes = extract_proposals(probs[0])
        labels = chip[..., 6]
        valid = labels < 4
        hist = np.bincount(instances[valid] * 4 + labels[valid], minlength=(len(boxes) + 1) * 4)
        crops = proposal_crops(before, after, boxes)
        path = work / record["file"]
        np.save(path, crops)
        samples.append((path, hist.reshape(-1, 4), dense_region_scores(probs, instances, len(boxes))))
        if i % 5 == 0 or i == len(records):
            print(f"Prepared validation regions {i}/{len(records)}", flush=True)
    return samples


class ObjectCrops:
    def __init__(self, cache, records, batch_size, seed):
        self.groups = [[] for _ in range(3)]
        self.rng = np.random.default_rng(seed)
        self.buffer = np.empty((batch_size, CHIP_SIZE, CHIP_SIZE, 7), np.uint8)
        for row in records:
            path = cache / "earthquake" / row["file"]
            for building in row["buildings"]:
                label = building["class_id"]
                if 1 <= label <= 3:
                    self.groups[label - 1].append((path, building["bounds"]))

    def batch(self):
        labels = self.rng.choice(3, size=len(self.buffer), p=(.60, .27, .13))
        for target, label in zip(self.buffer, labels):
            group = self.groups[label]
            path, box = group[self.rng.integers(len(group))]
            pre, post = read_pair(path)
            target[:] = transform_chip(building_crop(pre, post, box, self.rng), self.rng)
        return self.buffer, labels


def validation(model, samples, device, harmonize=False):
    calibrations = [(blend, Calibration(damage_thresholds=FINAL_THRESHOLDS)) for blend in (0., 0.30)]
    for path, hist, dense in samples:
        crops = np.load(path, mmap_mode="r")
        scores = score_crops(model, crops, device, harmonize=harmonize)
        for blend, acc in calibrations:
            acc.add_regions(hist, region_scores(scores, dense, blend))
    result = []
    for blend, acc in calibrations:
        thresholds, report = acc.fit()
        score = 0.25 * report["footprint"]["f1"] + 0.75 * report["damaged_macro_f1"]
        result.append({"blend": blend, "harmonize": harmonize, "thresholds": thresholds, "validation": report,
                       "selection_score": score})
    return max(result, key=lambda row: row["selection_score"])


def fit(args, device):
    started = time.monotonic()
    records = json.loads((args.cache / "earthquake.json").read_text(encoding="utf-8"))
    dense_model, state = load_model(args.footprint, device)
    samples = prepare_validation(args.cache, records["val"], dense_model, state, device)
    if args.init is None:
        model = RegionTransformer().to(device)
        for target, source in zip(model.pyramid, dense_model.pyramid):
            target.load_state_dict(source.state_dict())
    else:
        model, previous = load_region_model(args.init, device)
        print(f"Initialized regional model from epoch {previous['epoch']}", flush=True)
    del dense_model
    if args.calibrate:
        results = [validation(model, samples, device, harmonize=mode) for mode in (False, True)]
        best = max(results, key=lambda row: row["selection_score"])
        save_checkpoint(args.out / "region_model.pt", model, stage="building-regions",
                        epoch=previous["epoch"], footprint_checkpoint=str(args.footprint.resolve()),
                        initialized_from=str(args.init.resolve()), **best)
        write_json(args.out / "selection.json", {"selected": best, "candidates": results})
        print(json.dumps({k: v for k, v in best.items() if k != "validation"}), flush=True)
        return
    data = ObjectCrops(args.cache, records["train"], args.batch, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00065 if args.init is None else 0.00045,
                                  weight_decay=.035)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00006)
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history, best, best_epoch = [], -1., 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = torch.zeros((), device=device)
        steps = 0
        for _ in range(args.steps):
            array, labels = data.batch()
            pre, post, roi = tensor_inputs(array, device)
            labels = torch.from_numpy(labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(augment_image(pre, .30), augment_image(post, .30), roi)
                logits = logits.float()
                loss = F.cross_entropy(logits, labels, label_smoothing=.035)
                any_damage = logits[:, 1:].logsumexp(dim=1) - logits[:, 0]
                loss += .25 * F.binary_cross_entropy_with_logits(any_damage, (labels != 0).float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach()
            steps += 1
            if time.monotonic() - started >= args.seconds:
                break
        scheduler.step()
        loss = float(total) / steps
        if not np.isfinite(loss):
            raise RuntimeError("Non-finite region training loss")
        row = {"epoch": epoch, "steps": steps, "loss": loss}
        check = epoch in (1, args.epochs) or epoch % 3 == 0 or time.monotonic() - started >= args.seconds
        if check:
            scores = validation(model, samples, device)
            row.update(scores)
            if scores["selection_score"] > best:
                best, best_epoch = scores["selection_score"], epoch
                save_checkpoint(args.out / "region_model.pt", model, stage="building-regions",
                                footprint_checkpoint=str(args.footprint.resolve()), **row)
                write_json(args.out / "validation_best.json", row)
        row["elapsed_seconds"] = round(time.monotonic() - started, 1)
        history.append(row)
        write_json(args.out / "region_history.json", history)
        summary = {key: value for key, value in row.items() if key != "validation"}
        if check:
            for key in CLASSES[2:]:
                summary[key] = row["validation"]["classes"][key]["f1"]
        print(json.dumps(summary), flush=True)
        if time.monotonic() - started >= args.seconds or (epoch >= 24 and epoch - best_epoch >= 18):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DATA)
    parser.add_argument("--footprint", type=Path, default=RUN / "damage_model.pt")
    parser.add_argument("--init", type=Path)
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate a saved --init checkpoint on the validation split")
    parser.add_argument("--out", type=Path, default=RUN / "regions")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--seconds", type=int, default=1900)
    parser.add_argument("--seed", type=int, default=1262)
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch, args.seconds) <= 0:
        parser.error("epochs, steps, batch, and seconds must be positive")
    if args.calibrate and args.init is None:
        parser.error("--calibrate requires --init")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit(args, device)


if __name__ == "__main__":
    main()
