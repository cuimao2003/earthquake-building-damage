"""Sliding-window inference and split-level pixel metrics."""

import cv2
import numpy as np
import torch

from prepare import CLASSES, SIZE


DAMAGE_THRESHOLDS = np.array([
    .34, .40, .45, .50, .55, .60, .65, .70, .75, .80, .84, .87, .90, .92,
    .94, .95, .96, .97, .98, .985, .99, .993, .995, .997, .999,
])
def grid_positions(length, tile, stride):
    return sorted(set(range(0, length - tile + 1, stride)) | {length - tile})


@torch.no_grad()
def predict(model, before, after, device, batch=12, amp=False, stride=192):
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("Before and after images must be equally sized RGB arrays")
    height, width = before.shape[:2]
    extra = ((0, max(0, SIZE - height)), (0, max(0, SIZE - width)), (0, 0))
    before, after = (np.pad(im, extra, mode="edge") if height < SIZE or width < SIZE else im
                     for im in (before, after))
    h, w = before.shape[:2]
    origins = [(y, x) for y in grid_positions(h, SIZE, stride)
               for x in grid_positions(w, SIZE, stride)]
    # Smaller weights near the crop edges reduce visible seams.
    weights = 0.15 + 0.85 * np.outer(np.hanning(SIZE), np.hanning(SIZE)).astype(np.float32)
    total = np.zeros((4, h, w), np.float32)
    support = np.zeros((h, w), np.float32)
    model.eval()
    for offset in range(0, len(origins), batch):
        group = origins[offset:offset + batch]
        inputs = []
        for im in (before, after):
            stack = np.stack([im[y:y + SIZE, x:x + SIZE] for y, x in group])
            inputs.append(torch.from_numpy(np.ascontiguousarray(stack.transpose(0, 3, 1, 2)))
                          .to(device).float().div_(127.5).sub_(1.))
        with torch.cuda.amp.autocast(enabled=amp):
            predictions = model.probabilities(*inputs).cpu().numpy()
        for (y, x), probs in zip(group, predictions):
            total[:, y:y + SIZE, x:x + SIZE] += probs * weights[None]
            support[y:y + SIZE, x:x + SIZE] += weights
    return (total / support[None])[:, :height, :width]


def predict_map(model, before, after, device, batch=12, amp=False, views=1, sigma=0.,
                normalize=False):
    if views not in (1, 4):
        raise ValueError("Rotation averaging requires one or four views")
    probs = predict(model, before, after, device, batch, amp)
    for k in range(1, views):
        rotated = predict(model, np.rot90(before, k), np.rot90(after, k), device, batch, amp)
        probs += np.rot90(rotated, -k, axes=(1, 2))
    probs /= views
    if sigma:
        probs = smooth_damage(probs, sigma)
    return normalize_damage(probs) if normalize else probs


def smooth_damage(probs, sigma):
    '''Average nearby damage evidence while giving background less influence.'''
    mass = cv2.GaussianBlur(probs[0], (0, 0), sigma).clip(1e-12)
    return np.stack([probs[0], *(cv2.GaussianBlur(p * probs[0], (0, 0), sigma) / mass
                                  for p in probs[1:])])


def normalize_damage(probs):
    """Moderate scene-wide differences in the baseline damage scores."""
    building = probs[0] >= 0.5
    if building.sum() < 128:
        return probs
    prior = np.quantile(probs[2:, building], 0.985, axis=1).clip(.05, .95)
    scale = ((1 - prior) / prior) ** .55
    normalized = probs.copy()
    normalized[2:] *= scale[:, None, None]
    normalized[1:] /= normalized[1:].sum(axis=0, keepdims=True).clip(1e-12)
    return normalized


def classify(probs, thresholds):
    localize = probs[0] >= thresholds["footprint"]
    return grade(localize, probs, thresholds)


def grade(localize, probs, thresholds):
    pred = localize.astype(np.uint8)
    minor = localize & (probs[2] >= probs[3]) & (probs[2] >= thresholds["minor"])
    severe = localize & (probs[3] > probs[2]) & (probs[3] >= thresholds["severe"])
    pred[minor] = 2
    pred[severe] = 3
    return pred


def confusion(truth, pred):
    keep = truth < len(CLASSES)
    codes = truth[keep].astype(np.int16) * len(CLASSES) + pred[keep]
    return np.bincount(codes, minlength=16).reshape(4, 4)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.


def metrics(matrix):
    true = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = matrix.diagonal()
    rows = {}
    for i, name in enumerate(CLASSES):
        rows[name] = {"precision": ratio(correct[i], predicted[i]),
                      "recall": ratio(correct[i], true[i]),
                      "f1": ratio(2 * correct[i], true[i] + predicted[i]),
                      "iou": ratio(correct[i], true[i] + predicted[i] - correct[i]),
                      "support": int(true[i]), "predicted": int(predicted[i])}
    tp = matrix[1:, 1:].sum()
    fp = matrix[0, 1:].sum()
    fn = matrix[1:, 0].sum()
    total = int(matrix.sum())
    acc = ratio(correct.sum(), total)
    # Float multiplication avoids overflow on larger collections of pixels.
    chance = ratio(np.dot(true.astype(np.float64), predicted), float(total) ** 2)
    damage_tp = matrix[2:, 2:].sum()
    damage_predictions = predicted[2:].sum()
    damage_truth = true[2:].sum()
    return {
        "confusion": matrix.tolist(),
        "class_order": list(CLASSES),
        "classes": rows,
        "accuracy": acc,
        "kappa": ratio(acc - chance, 1 - chance),
        "footprint": {"f1": ratio(2 * tp, 2 * tp + fp + fn),
                      "iou": ratio(tp, tp + fp + fn),
                      "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn)},
        "any_damage": {"f1": ratio(2 * damage_tp, damage_predictions + damage_truth),
                       "precision": ratio(damage_tp, damage_predictions),
                       "recall": ratio(damage_tp, damage_truth)},
        "building_macro_f1": float(np.mean([rows[key]["f1"] for key in CLASSES[1:]])),
        "damaged_macro_f1": float(np.mean([rows[key]["f1"] for key in CLASSES[2:]])),
    }


class Calibration:
    def __init__(self, footprint_threshold=0.5, damage_thresholds=DAMAGE_THRESHOLDS):
        self.footprint_threshold = footprint_threshold
        self.damage_thresholds = damage_thresholds
        self.counts = np.zeros((2, 4, len(damage_thresholds) + 1), np.int64)
        self.missed = np.zeros(4, np.int64)

    def add(self, truth, probs):
        keep = truth < 4
        localized = (probs[0] >= self.footprint_threshold) & keep
        self.missed += np.bincount(truth[keep & ~localized], minlength=4)
        labels = truth[localized].astype(np.int64)
        minor, severe = probs[2][localized], probs[3][localized]
        candidate = severe > minor
        grid = self.damage_thresholds.astype(minor.dtype, copy=False)
        buckets = np.searchsorted(grid, np.maximum(minor, severe), side="right")
        count = self.counts.shape[-1]
        codes = (4 * candidate.astype(np.int64) + labels) * count + buckets
        self.counts += np.bincount(codes, minlength=self.counts.size).reshape(self.counts.shape)

    def add_regions(self, truth_counts, probs):
        self.missed += truth_counts[0]
        minor, severe = probs[1:, 1:].T
        candidate = severe > minor
        grid = self.damage_thresholds.astype(minor.dtype, copy=False)
        buckets = np.searchsorted(grid, np.maximum(minor, severe), side="right")
        count = self.counts.shape[-1]
        codes = candidate.astype(np.int64) * count + buckets
        for label in range(4):
            weighted = np.bincount(codes, weights=truth_counts[1:, label], minlength=2 * count)
            self.counts[:, label] += weighted.reshape(2, count).astype(np.int64)

    def fit(self):
        true = self.counts.sum(axis=(0, 2)) + self.missed
        remaining = self.counts.sum(axis=(0, 2))
        matrix = np.zeros((4, 4), np.int64)
        matrix[:, 0] = self.missed
        thresholds = {"footprint": self.footprint_threshold}
        for slot, name in enumerate(("minor", "severe")):
            label = slot + 2
            positives = np.flip(np.cumsum(np.flip(self.counts[slot], axis=-1), axis=-1), axis=-1)[:, 1:]
            f1 = 2 * positives[label] / np.maximum(1, true[label] + positives.sum(axis=0))
            if np.max(f1) > 0:
                index = int(np.flatnonzero(f1 == max(f1))[-1])
            else:
                index = int(np.argmin(np.abs(self.damage_thresholds - 0.50)))
            thresholds[name] = float(self.damage_thresholds[index])
            matrix[:, label] = positives[:, index]
            remaining = remaining - matrix[:, label]
        matrix[:, 1] = remaining
        return thresholds, metrics(matrix)


def validation(model, cache, records, device, batch, amp):
    calibration = Calibration()
    for record in records:
        chip = np.load(cache / "earthquake" / record["file"], mmap_mode="r")
        probs = predict(model, chip[..., :3], chip[..., 3:6], device, batch, amp)
        calibration.add(chip[..., 6], probs)
    return calibration.fit()
