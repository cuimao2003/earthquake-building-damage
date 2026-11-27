"""Train and evaluate the paired proposal model on xBD earthquake imagery."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from classifier import odds, predict_grade
from detector import box_iou, match
from dino import load_dino
from features import WEIGHTS as REFERENCE_ENCODER, align, jitter_bounds, read_rgb
from infer_retrain import (SelectedGrader, evaluate, ground_truth, paired_images, predict_pair,
                           tune_automatic_thresholds)
from inventory import ROOT, overlap, scan, write_json
from roi_transformer import PATCH_SIZE, RoofTransformer, encode_boxes, encode_roofs, probabilities, training_loss
from swin_locator import TILE, load_swin, swin_boxes


MANIFEST = ROOT / "data/manifest.json"
WEIGHTS = ROOT / "weights/dinov2_vitb14_pretrain.pth"
DETECTOR = ROOT / "run/swin_loc_mix/detector.pt"
CACHE = ROOT / "data/proposal_transformer"
NAME = "proposal_transformer"
REFERENCE = ROOT / "run/mexico_large_selection"


def digest(path):
    fingerprint = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    return fingerprint.hexdigest()


def index(args):
    return json.loads(args.manifest.read_bytes())


def detector_boxes(detector, state, image, device, floor=None):
    threshold = state["threshold"] if floor is None else floor
    boxes, _ = swin_boxes(detector, image, device, threshold, tile=state.get("tile_size", TILE))
    return boxes


def subset(ids, cap, rng):
    if len(ids) <= cap:
        return ids
    return np.sort(rng.choice(ids, cap, replace=False))


def training_boxes(truth, grades, proposed, shape, rng):
    active = np.r_[subset(np.flatnonzero(grades == 0), 46, rng), np.flatnonzero(grades > 0)]
    positive = np.flatnonzero(grades > 0)
    jittered, jittered_grades = [], []
    for i in positive:
        for _ in range(3):
            candidate = np.asarray(jitter_bounds(truth[i], rng, shape), np.float32)
            if box_iou(candidate[None], truth[i:i + 1])[0, 0] >= .50:
                jittered.append(candidate)
                jittered_grades.append(grades[i])
    associated, truth_ids = match(proposed, truth)
    proposed_grade = np.full(len(proposed), -2, np.int8)
    proposed_grade[associated] = np.where(grades[truth_ids] >= 0, grades[truth_ids], -2)
    if len(truth) and np.any(grades >= 0):
        isolated = box_iou(proposed, truth).max(1) < .20
        proposed_grade[isolated] = -1
    chosen = np.r_[subset(np.flatnonzero(proposed_grade == 0), 26, rng),
                   np.flatnonzero(proposed_grade > 0), subset(np.flatnonzero(proposed_grade == -1), 24, rng)]
    boxes = np.concatenate((truth[active], np.asarray(jittered, np.float32).reshape(-1, 4), proposed[chosen]))
    labels = np.concatenate((grades[active], np.asarray(jittered_grades, np.int8), proposed_grade[chosen]))
    source = np.repeat(np.arange(3, dtype=np.int8), [len(active), len(jittered), len(chosen)])
    return boxes, labels, source


def atomic_arrays(path, **arrays):
    pending = path.with_name(path.stem + ".building.npz")
    np.savez(pending, **arrays)
    pending.replace(path)


def prepare(args, device):
    header = {"format": 1, "manifest_sha256": digest(args.manifest), "encoder_sha256": digest(args.weights),
              "detector_sha256": digest(args.detector), "pilot_scenes": args.pilot_scenes, "patch_size": PATCH_SIZE,
              "registration": args.registration}
    if args.detector_floor is not None:
        header["detector_floor"] = args.detector_floor
    args.cache.mkdir(parents=True, exist_ok=True)
    stamp = args.cache / "setup.json"
    if stamp.is_file() and json.loads(stamp.read_bytes()) != header:
        raise ValueError("Cache was prepared with different inputs or options. Use another --cache directory.")
    write_json(stamp, header)
    rows = index(args)
    detector, state = load_swin(args.detector, device)
    encoder = load_dino(args.weights, device)
    for split in ("train", "val"):
        records = rows[split]
        if args.pilot_scenes:
            records = sorted(records, key=lambda row: -sum(row["counts"].get(k, 0)
                             for k in ("minor-damage", "major-damage", "destroyed")))[:args.pilot_scenes]
        folder = args.cache / split
        folder.mkdir(exist_ok=True)
        write_json(folder / "chips.json", [row["chip"] for row in records])
        for n, row in enumerate(records, 1):
            output = folder / f"{row['chip']}.npz"
            if output.is_file():
                continue
            before, after = paired_images(row)
            proposed = detector_boxes(detector, state, before, device, args.detector_floor)
            truth, grades = ground_truth(row)
            registered = align(before, after, args.registration)[0]
            if split == "train":
                rng = np.random.default_rng(sum(map(ord, row["chip"])) * 9023)
                boxes, target, kind = training_boxes(truth, grades, proposed, before.shape, rng)
                data = dict(target=target, kind=kind)
            else:
                boxes = proposed
                data = dict(boxes=boxes, truth=truth, grades=grades)
            features, roi = encode_boxes(encoder, before, registered, boxes, device, args.encode_batch)
            atomic_arrays(output, features=features, roi=roi, **data)
            if n % 10 == 0 or n == len(records):
                print(f"{split} cache: {n}/{len(records)} scenes", flush=True)
    if args.morocco:
        auxiliary_stamp = args.cache / "morocco_source.json"
        identity = {"source_sha256": digest(args.morocco)}
        if (args.cache / "morocco.npz").is_file() and auxiliary_stamp.is_file():
            if json.loads(auxiliary_stamp.read_bytes()) != identity:
                raise ValueError("Morocco source differs from this cache. Use another --cache directory.")
            return
        with np.load(args.morocco, allow_pickle=False) as data:
            rgb, roi, labels = data["rgb"], data["roi"], data["grade"]
        if set(np.unique(labels)) != {1, 2}:
            raise ValueError("The Morocco auxiliary cache must have damaged and destroyed grades")
        features, weights = [], []
        for start in range(0, len(labels), args.encode_batch):
            work = slice(start, start + args.encode_batch)
            tokens, mask = encode_roofs(encoder, rgb[work], roi[work], device)
            features.append(tokens)
            weights.append(mask)
        atomic_arrays(args.cache / "morocco.npz", features=np.concatenate(features), roi=np.concatenate(weights),
                      target=(labels == 2).astype(np.float32))
        write_json(auxiliary_stamp, identity)
        print(f"Morocco auxiliary cache: {len(labels)} labeled pairs", flush=True)


def scene_files(cache, split):
    folder = cache / split
    for key in json.loads((folder / "chips.json").read_bytes()):
        yield folder / f"{key}.npz"


def load_split(cache, split):
    for path in scene_files(cache, split):
        with np.load(path, allow_pickle=False) as arrays:
            yield {name: arrays[name] for name in arrays.files}


def training_data(cache, manual_only):
    size = 0
    for path in scene_files(cache, "train"):
        with np.load(path, allow_pickle=False) as arrays:
            size += np.count_nonzero(arrays["kind"] == 0) if manual_only else len(arrays["target"])
    start, data = 0, {}
    for scene in load_split(cache, "train"):
        keep = scene["kind"] == 0 if manual_only else slice(None)
        stop = start + len(scene["target"][keep])
        for name in ("features", "roi", "target"):
            values = scene[name]
            if name not in data:
                data[name] = np.empty((size, *values.shape[1:]), values.dtype)
            data[name][start:stop] = values[keep]
        start = stop
    if not size:
        raise ValueError("Training cache is empty")
    return data


def tensor(array, device):
    return torch.from_numpy(array).to(device)


@torch.no_grad()
def score(head, features, roi, device, batch=128):
    outputs = []
    head.eval()
    for start in range(0, len(features), batch):
        end = start + batch
        # The PyTorch 1.12 Transformer inference fast path requires matching dtypes.
        logits = head(tensor(features[start:end], device), tensor(roi[start:end], device).float())
        p, q = probabilities(logits)
        outputs.append(np.column_stack((p.cpu().numpy(), q.cpu().numpy())))
    return np.concatenate(outputs) if outputs else np.zeros((0, 4), np.float32)


def calibrate(head, validation, device, has_background):
    scored = [score(head, d["features"], d["roi"], device) for d in validation]
    candidates = []
    thresholds = (0., .30, .50, .65, .80) if has_background else (0.,)
    for valid_limit in thresholds:
        values, labels = [], []
        missed = np.zeros(3, np.int64)
        matches = proposed = total = 0
        for row, result in zip(validation, scored):
            keep = result[:, 0] >= valid_limit
            boxes, grades = row["boxes"][keep], row["grades"]
            p, t = match(boxes, row["truth"])
            assigned = np.full(len(boxes), -1, np.int8)
            assigned[p] = np.where(grades[t] >= 0, grades[t], -2)
            known = assigned >= -1
            values.append(result[keep, 1:][known])
            labels.append(assigned[known])
            remaining = np.ones(len(grades), bool)
            remaining[t] = False
            missed += np.bincount(grades[remaining & (grades >= 0)], minlength=3)
            matches += len(p)
            proposed += len(boxes)
            total += len(grades)
        scores, targets = np.concatenate(values), np.concatenate(labels)
        if not len(scores):
            continue
        setting = tune_automatic_thresholds(scores, targets, missed)
        support = np.bincount(targets[targets >= 0], minlength=3) + missed
        setting.update(validity_threshold=valid_limit, box_f1=2 * matches / max(1, proposed + total),
                       localization_by_grade={"matched": (support - missed).tolist(), "annotated": support.tolist()})
        candidates.append(setting)
    return max(candidates, key=lambda row: (row["validation"]["macro_f1"],
                                            row["validation"]["damaged_macro_f1"]))


def save_head(path, head, setup):
    pending = path.with_suffix(".part")
    torch.save({"head": head.state_dict(), **setup}, pending)
    pending.replace(path)


def fit(args, device):
    stamp = json.loads((args.cache / "setup.json").read_bytes())
    if stamp["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Training requires the manifest that was used to build the cache")
    data = training_data(args.cache, args.manual_only)
    validation = list(load_split(args.cache, "val"))
    groups = {k: np.flatnonzero(data["target"] == k) for k in (-1, 0, 1, 2)}
    if any(not len(groups[k]) for k in (0, 1, 2)):
        raise ValueError("Training cache needs labeled intact, light and heavy damage buildings")
    budget = {0: 28, 1: 14, 2: 10}
    has_background = bool(len(groups[-1]))
    if has_background:
        budget[-1] = 12
    width = data["features"].shape[-1]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    head = RoofTransformer(width, args.variant).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=.00035, weight_decay=.045)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=.00004)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    auxiliary = auxiliary_identity = None
    if args.use_morocco:
        auxiliary_identity = json.loads((args.cache / "morocco_source.json").read_bytes())
        with np.load(args.cache / "morocco.npz", allow_pickle=False) as z:
            auxiliary = dict(z)
    best, history = -1., []
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "head.pt").exists():
        raise ValueError("Output already contains a trained head. Use a different --out directory.")
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.
        steps = 0
        for _ in range(args.steps):
            ids = np.concatenate([rng.choice(groups[k], count, replace=True) for k, count in budget.items()])
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = head(tensor(data["features"][ids], device), tensor(data["roi"][ids], device).float())
                loss = training_loss(logits.float(), tensor(data["target"][ids], device), has_background)
                if auxiliary is not None:
                    ids = rng.integers(0, len(auxiliary["target"]), 16)
                    extra = head(tensor(auxiliary["features"][ids], device), tensor(auxiliary["roi"][ids], device).float())
                    loss += .20 * F.binary_cross_entropy_with_logits(extra[:, 3].float(), tensor(auxiliary["target"][ids], device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss on epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total += float(loss.detach())
            steps += 1
            if time.monotonic() - start >= args.seconds:
                break
        schedule.step()
        row = {"epoch": epoch, "loss": total / steps, "elapsed_seconds": time.monotonic() - start}
        stop = time.monotonic() - start >= args.seconds
        if epoch == 1 or epoch % 4 == 0 or epoch == args.epochs or stop:
            chosen = calibrate(head, validation, device, has_background)
            row["validation"] = chosen
            macro = chosen["validation"]["macro_f1"]
            if macro > best:
                best = macro
                save_head(args.out / "head.pt", head, {"width": width, "variant": args.variant, "epoch": epoch,
                          "cache": stamp, "calibration": chosen, "manual_only": args.manual_only,
                          "auxiliary": auxiliary_identity, "seed": args.seed})
                write_json(args.out / "validation.json", chosen)
            print(f"Epoch {epoch}: loss {row['loss']:.4f}, automatic validation macro F1 {macro:.4f}", flush=True)
        history.append(row)
        write_json(args.out / "history.json", history)
        if stop:
            break


def fusion_scores(reference, proposed, weight):
    log_odds = (1 - weight) * odds(reference) + weight * odds(proposed)
    logits = np.column_stack((np.zeros(len(log_odds)), log_odds))
    scores = np.exp(logits - logits.max(1, keepdims=True))
    return scores / scores.sum(1, keepdims=True)


def reference_files(folder):
    config = folder / "selection.json"
    selected = json.loads(config.read_bytes())["selected"]
    files = {config, folder / "automatic_thresholds.json"}
    for key in ("minor", "major"):
        branch = selected[key]
        directory = Path(branch["model"])
        encoder = Path(branch["encoder"]) if branch["encoder"] else REFERENCE_ENCODER
        files.update({directory / "head.pt", directory / "selection.json", encoder})
        texture = directory / "texture.joblib"
        if texture.is_file():
            files.add(texture)
    return {str(path.resolve()): digest(path) for path in sorted(files)}


class ProposalGrader:
    def __init__(self, args, device):
        self.device, self.batch = device, args.encode_batch
        state = torch.load(args.out / "head.pt", map_location="cpu")
        self.settings = state["calibration"]
        self.registration = state["cache"]["registration"]
        self.detector_floor = state["cache"].get("detector_floor")
        if state["cache"]["encoder_sha256"] != digest(args.weights) or state["cache"]["detector_sha256"] != digest(args.detector):
            raise ValueError("Encoder or detector differs from the locked training inputs")
        self.encoder = load_dino(args.weights, device)
        self.detector, self.detector_state = load_swin(args.detector, device)
        self.head = RoofTransformer(state["width"], state["variant"]).to(device)
        self.head.load_state_dict(state["head"])
        self.previous = None
        self.reference = None
        if args.fusion:
            blend = json.loads(args.fusion.read_bytes())
            if blend["candidate_sha256"] != digest(args.out / "head.pt"):
                raise ValueError("Fusion configuration was calibrated against a different candidate")
            folder = Path(blend["reference"])
            if blend["reference_files"] != reference_files(folder):
                raise ValueError("The reference inputs have changed since fusion calibration")
            self.reference = SelectedGrader(folder, device)
            self.fusion_weight = np.asarray(blend["selected"]["weight"])
            self.settings = blend["selected"]

    def raw_scores(self, pre, post, boxes):
        registered = align(pre, post, self.registration)[0]
        visual, roi = encode_boxes(self.encoder, pre, registered, boxes, self.device, self.batch)
        return score(self.head, visual, roi, self.device)

    def scores(self, pre, post, boxes):
        results = self.raw_scores(pre, post, boxes)
        if self.reference is not None and len(boxes):
            _, reference = self.reference.classify(pre, post, boxes)
            results[:, 1:] = fusion_scores(reference, results[:, 1:], self.fusion_weight)
        return results

    def propose(self, pre, post):
        boxes = detector_boxes(self.detector, self.detector_state, pre, self.device, self.detector_floor)
        results = self.scores(pre, post, boxes)
        keep = results[:, 0] >= self.settings["validity_threshold"]
        retained = boxes[keep]
        self.previous = retained, results[keep, 1:]
        return retained

    def classify(self, pre, post, boxes):
        probs = (self.previous[1] if self.previous is not None and self.previous[0] is boxes
                 else self.scores(pre, post, boxes)[:, 1:])
        return predict_grade(probs, np.asarray(self.settings["thresholds"])), probs


def independent_rows(args, destination, groups):
    from evaluate_test_folder import collect

    candidates, history = collect(args.test_folder.resolve(), destination)
    sources = {row["source"] for row in candidates}
    lookup = {(r["source"], r["chip"]): r for source in sources for r in scan(Path(source), "external")}
    registered = json.loads(MANIFEST.read_bytes())
    seen_rows = [r for inventory in (groups, registered) for group in inventory.values() for r in group]
    seen_chips = {r["chip"] for r in seen_rows}
    seen_sources = {r["source"] for r in seen_rows}
    references = [r for source in seen_sources for r in scan(Path(source), "previous") if r["chip"] in seen_chips]
    seen_uids = set().union(*(r["uids"] for r in references))
    archived = ROOT.parent / "cache/earthquake.json"
    archive = json.loads(archived.read_bytes()) if archived.is_file() else {}
    old_hashes = {h for rows in archive.values() if isinstance(rows, list)
                  for row in rows if isinstance(row, dict) for h in row.get("hashes", [])}
    kept, excluded = [], []
    for row in candidates:
        chip = row["chip"]
        target = lookup[(row["source"], chip)]
        reason = None
        if chip in seen_chips or history[chip] != "unlisted" or seen_uids & target["uids"]:
            reason = "previous_chip_or_buildings"
        elif any(digest(Path(row["source"]) / "images" / f"{chip}_{when}_disaster.png") in old_hashes
                 for when in ("pre", "post")):
            reason = "previous_image_hash"
        elif target["extent"] is None:
            reason = "geographic_overlap_not_verifiable"
        elif any(overlap(target["extent"], r["extent"]) > .003 for r in references):
            reason = "geographic_overlap"
        if reason:
            excluded.append({"chip": chip, "reason": reason})
        else:
            kept.append(row)
    write_json(destination / "independence_audit.json", {"evaluated": len(kept), "excluded": excluded})
    if not kept:
        raise ValueError("No independent earthquake pairs passed the audit. The audit lists the exclusions.")
    counts = Counter()
    for row in kept:
        counts.update(row["counts"])
    if not (counts["no-damage"] and counts["minor-damage"] and (counts["major-damage"] or counts["destroyed"])):
        raise ValueError("Independent evaluation needs manual annotations in all three damage grades")
    return kept


def test(args, device):
    groups = index(args)
    checkpoint = torch.load(args.out / "head.pt", map_location="cpu")
    if checkpoint["cache"]["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Evaluation requires the same manifest that was used for training")
    if checkpoint["cache"]["pilot_scenes"] and not args.regression:
        raise ValueError("Use a full training run for independent evaluation")
    if args.result.exists() and any(args.result.iterdir()):
        raise ValueError("Use a fresh --result directory")
    args.result.mkdir(parents=True, exist_ok=True)
    if args.regression:
        rows = groups["test"]
        if args.test_folder:
            from evaluate_test_folder import collect

            rows, _ = collect(args.test_folder.resolve(), args.result)
    else:
        rows = independent_rows(args, args.result, groups)
    grader = ProposalGrader(args, device)
    protocol = {"evaluation": "regression" if args.regression else "screened_external",
                "training_run": str(args.out.resolve()), "scenes": len(rows),
                "checkpoint_sha256": digest(args.out / "head.pt")}
    if args.fusion:
        protocol["fusion_sha256"] = digest(args.fusion)
    write_json(args.result / "protocol.json", protocol)
    name = NAME + "_fusion" if args.fusion else NAME
    evaluate(rows, grader, name, None, args.result, known_boxes=args.known_boxes, pixel_audit=True,
             boxes_from=grader.propose, registration_method=grader.registration)


def audit_location(args, device):
    model, state = load_swin(args.detector, device)
    cutoffs = sorted({.12, .16, .24, .36, .48, .60, .72, .83, state["threshold"]})
    totals = np.zeros((len(cutoffs), 3), np.int64)
    attainable = np.zeros_like(totals)
    support = np.zeros(3, np.int64)
    records = index(args)["val"]
    for row in records:
        chip = row["chip"]
        pre = read_rgb(Path(row["source"]) / "images" / f"{chip}_pre_disaster.png")
        truth, grades = ground_truth(row)
        support += np.bincount(grades[grades >= 0], minlength=3)
        proposed, scores = swin_boxes(model, pre, device, cutoffs[0], tile=state.get("tile_size", TILE))
        for i, limit in enumerate(cutoffs):
            boxes = proposed[scores >= limit]
            p, _ = match(boxes, truth)
            totals[i] += [len(p), len(boxes), len(truth)]
            for grade in range(3):
                _, matched = match(boxes, truth[grades == grade])
                attainable[i, grade] += len(matched)
    results = []
    for limit, (matches, boxes, truth), found in zip(cutoffs, totals, attainable):
        upper = 2 * found / np.maximum(1, support + found)
        results.append({"detector_threshold": float(limit), "proposals": int(boxes),
                        "box_f1": float(2 * matches / max(1, boxes + truth)),
                        "per_grade_match_upper_bound": found.tolist(), "macro_f1_upper_bound": float(upper.mean())})
    args.result.mkdir(parents=True, exist_ok=True)
    write_json(args.result / "location_audit.json", {"scenes": len(records), "support": support.tolist(),
                 "default_threshold": state["threshold"], "thresholds": results})


def fit_fusion(args, device):
    if args.fusion:
        raise ValueError("Fusion fitting requires an unfused candidate")
    checkpoint = torch.load(args.out / "head.pt", map_location="cpu")
    stamp = json.loads((args.cache / "setup.json").read_bytes())
    if stamp != checkpoint["cache"] or stamp["pilot_scenes"] or stamp.get("detector_floor") is not None:
        raise ValueError("Fusion fitting requires the candidate's full cache with default locator settings")
    if stamp["manifest_sha256"] != digest(args.manifest) or stamp["detector_sha256"] != digest(DETECTOR):
        raise ValueError("Fusion requires the manifest and Swin detector used by the reference")
    reference = SelectedGrader(args.reference, device)
    if reference.calibrated_localizer != "swin_mix":
        raise ValueError("The reference must be calibrated for the same Swin detector")
    head = RoofTransformer(checkpoint["width"], checkpoint["variant"]).to(device)
    head.load_state_dict(checkpoint["head"])
    fixed, trainable, labels = [], [], []
    missed = np.zeros(3, np.int64)
    records = index(args)["val"]
    if [r["chip"] for r in records] != json.loads((args.cache / "val/chips.json").read_bytes()):
        raise ValueError("Validation scene order differs from the locked cache")
    for i, (row, cached) in enumerate(zip(records, load_split(args.cache, "val")), 1):
        boxes, truth, grades = cached["boxes"], cached["truth"], cached["grades"]
        p, t = match(boxes, truth)
        target = np.full(len(boxes), -1, np.int8)
        target[p] = np.where(grades[t] >= 0, grades[t], -2)
        known = target >= -1
        remaining = np.ones(len(grades), bool)
        remaining[t] = False
        missed += np.bincount(grades[remaining & (grades >= 0)], minlength=3)
        _, baseline = reference.classify(*paired_images(row), boxes)
        fixed.append(baseline[known])
        trainable.append(score(head, cached["features"], cached["roi"], device)[known, 1:])
        labels.append(target[known])
        if i % 5 == 0 or i == len(records):
            print(f"Fusion validation {i}/{len(records)} scenes", flush=True)
    baseline, proposed, targets = map(np.concatenate, (fixed, trainable, labels))
    trials = []
    for light in (0., .20, .50, .80, 1.):
        for heavy in (0., .20, .50, .80, 1.):
            weight = np.asarray([light, heavy])
            result = tune_automatic_thresholds(fusion_scores(baseline, proposed, weight), targets, missed)
            trials.append({"weight": weight.tolist(), "validity_threshold": 0., **result})
    best = max(trials, key=lambda row: row["validation"]["macro_f1"])
    args.result.mkdir(parents=True, exist_ok=True)
    write_json(args.result / "fusion.json", {"candidate_sha256": digest(args.out / "head.pt"),
               "reference": str(args.reference.resolve()), "reference_files": reference_files(args.reference),
               "scenes": len(records), "selected": best, "trials": trials})
    print(f"Fusion selected weight {best['weight']}, validation macro F1 {best['validation']['macro_f1']:.4f}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("prepare", "train", "test", "image", "audit-location", "fuse"))
    cli.add_argument("--manifest", type=Path, default=MANIFEST)
    cli.add_argument("--weights", type=Path, default=WEIGHTS)
    cli.add_argument("--detector", type=Path, default=DETECTOR)
    cli.add_argument("--detector-floor", type=float, help="Override the proposal threshold during cache preparation")
    cli.add_argument("--reference", type=Path, default=REFERENCE)
    cli.add_argument("--fusion", type=Path, help="A fusion.json selected on validation data")
    cli.add_argument("--cache", type=Path, default=CACHE)
    cli.add_argument("--out", type=Path, default=ROOT / "run/proposal_transformer")
    cli.add_argument("--registration", choices=("phase", "keypoint"), default="keypoint")
    cli.add_argument("--encode-batch", type=int, default=24)
    cli.add_argument("--pilot-scenes", type=int, default=0)
    cli.add_argument("--morocco", type=Path)
    cli.add_argument("--use-morocco", action="store_true")
    cli.add_argument("--variant", choices=("pooled", "attention"), default="attention")
    cli.add_argument("--manual-only", action="store_true")
    cli.add_argument("--epochs", type=int, default=32)
    cli.add_argument("--steps", type=int, default=65)
    cli.add_argument("--seconds", type=int, default=1200)
    cli.add_argument("--seed", type=int, default=341561)
    cli.add_argument("--result", type=Path)
    cli.add_argument("--test-folder", type=Path)
    cli.add_argument("--regression", action="store_true")
    cli.add_argument("--known-boxes", action="store_true", help="Also diagnose grading with human boxes in test mode")
    cli.add_argument("--pre", type=Path)
    cli.add_argument("--post", type=Path)
    args = cli.parse_args()
    if min(args.encode_batch, args.epochs, args.steps, args.seconds) < 1 or args.pilot_scenes < 0:
        cli.error("Batch, epochs, steps and time must be positive")
    if args.detector_floor is not None and (args.action != "prepare" or not .05 <= args.detector_floor <= 1):
        cli.error("--detector-floor is a preparation option in the range .05 to 1")
    if args.action in ("test", "image", "audit-location", "fuse") and args.result is None:
        cli.error("Pass --result for evaluation, image, fusion or location-audit output")
    if args.fusion and args.action not in ("test", "image"):
        cli.error("--fusion can be used in test or image mode")
    if args.action == "test" and not (args.regression or args.test_folder):
        cli.error("Use --test-folder for an independent test, or --regression for the existing holdout")
    if args.action == "image" and (args.pre is None or args.post is None):
        cli.error("Image inference requires --pre and --post")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.action == "prepare":
        prepare(args, device)
    elif args.action == "train":
        fit(args, device)
    elif args.action == "test":
        test(args, device)
    elif args.action == "audit-location":
        audit_location(args, device)
    elif args.action == "fuse":
        fit_fusion(args, device)
    else:
        args.result.mkdir(parents=True, exist_ok=True)
        grader = ProposalGrader(args, device)
        name = NAME + "_fusion" if args.fusion else NAME
        predict_pair(read_rgb(args.pre), read_rgb(args.post), grader, name, None, args.result,
                     boxes_from=grader.propose, registration_method=grader.registration)


if __name__ == "__main__":
    main()
