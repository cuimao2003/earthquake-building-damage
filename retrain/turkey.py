"""Prepare the manually reviewed OCD-BDA Türkiye earthquake release."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import cv2
import numpy as np
import torch

from dino import load_dino
from features import WEIGHTS, STORE, embed, patch_pair, read_rgb, signal
from inventory import ROOT, write_json

SOURCE = ROOT / "data/external/turkey_ocd_bda.zip"
DATA = ROOT / "data/turkey"
# The original release reports Grade C = 3,287, B = 701, A = 3,262.
# Its class directories are 0 = 3,284, 1 = 700, 2 = 3,254 (v1.2 data sheet).
GRADES = {"2": 0, "1": 1, "0": 2}


class ArchiveParts(io.RawIOBase):
    """Use the available archive prefix and its separately fetched ZIP index."""

    size = 1020840303

    def __init__(self):
        self.part = SOURCE.with_suffix(".part").open("rb")
        self.tail = (SOURCE.parent / "turkey_tail.bin").read_bytes()
        self.offset = 0
        self.cut = self.size - len(self.tail)

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        self.offset = [0, self.offset, self.size][whence] + offset
        return self.offset

    def read(self, count=-1):
        if self.offset < self.cut:
            self.part.seek(self.offset)
            data = self.part.read(count)
        else:
            local = self.offset - self.cut
            data = self.tail[local:] if count < 0 else self.tail[local:local + count]
        self.offset += len(data)
        return data

    def close(self):
        self.part.close()
        super().close()


def unpack():
    DATA.mkdir(parents=True, exist_ok=True)
    written = Counter()
    reader = SOURCE.open("rb") if SOURCE.exists() else ArchiveParts()
    with reader, zipfile.ZipFile(reader) as package:
        for member in package.infolist():
            parts = PurePosixPath(member.filename)
            if member.is_dir() or len(parts.parts) < 2:
                continue
            path = "/" + str(parts)
            if "/IMG/post/" in path and parts.suffix.lower() == ".jpg":
                kind = "post"
            elif "/IMG/pro/" in path and parts.suffix.lower() == ".jpg":
                kind = "pre"
            elif "/saved/" in path and parts.suffix.lower() == ".json":
                kind = "labels"
            else:
                continue
            stem = parts.stem
            if stem.startswith("post"):
                stem = stem[4:]
            elif stem.startswith("pro"):
                stem = stem[3:]
            if not stem.isdecimal():
                raise ValueError(f"Unrecognized frame number: {parts.name}")
            output = DATA / kind / f"{int(stem):02d}{parts.suffix}"
            output.parent.mkdir(exist_ok=True)
            if not output.exists():
                stage = output.with_suffix(".tmp")
                stage.write_bytes(package.read(member))
                stage.replace(output)
            written[kind] += 1
    if written != {"pre": 60, "post": 60, "labels": 60}:
        raise ValueError(f"Incomplete image inventory {written}")
    print(dict(written), flush=True)


def load_regions(num):
    labels = json.loads((DATA / "labels" / f"{num:02d}.json").read_bytes())
    if set(labels) != set(GRADES):
        raise ValueError(f"Unknown labels: {num}: {list(labels)}")
    boxes = []
    for raw, items in labels.items():
        for box in items:
            bounds = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
            if (bounds[2] - bounds[0]) >= 2 and (bounds[3] - bounds[1]) >= 2:
                boxes.append((bounds, GRADES[raw]))
    return boxes


def imprint(path, size):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    f = size / max(image.shape)
    shape = (int(round(image.shape[1] * f)), int(round(image.shape[0] * f)))
    image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
    key, desc = cv2.SIFT_create(nfeatures=420).detectAndCompute(image, None)
    points = np.float32([k.pt for k in key])
    return points, desc, image.shape


def compare(first, second):
    p1, f1, shape1 = first
    p2, f2, shape2 = second
    if f1 is None or f2 is None or len(f1) < 18 or len(f2) < 18:
        return 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(f1, f2, k=2)
    selected = [m for m, n in knn if m.distance < .70 * n.distance]
    if len(selected) < 15:
        return 0
    a = np.float32([p1[m.queryIdx] for m in selected])
    b = np.float32([p2[m.trainIdx] for m in selected])
    affine, status = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3.)
    if affine is None or status is None:
        return 0
    inlier = status.ravel().astype(bool)
    if inlier.sum() < 14 or inlier.mean() < .50 or np.ptp(a[inlier], axis=0).min() < .10 * min(shape1):
        return 0
    scale = np.hypot(affine[0, 0], affine[0, 1])
    return int(inlier.sum()) if .45 < scale < 2.2 else 0


def scene_groups():
    photos = [imprint(DATA / "pre" / f"{num:02d}.jpg", 640) for num in range(1, 61)]
    groups = [{i} for i in range(60)]
    owners = list(range(60))
    overlapping = []
    for i, first in enumerate(photos):
        for j, second in enumerate(photos[:i]):
            if owners[i] == owners[j]:
                continue
            amount = compare(first, second)
            if amount:
                overlapping.append([i + 1, j + 1, amount])
                a, b = owners[i], owners[j]
                groups[a].update(groups[b])
                for k in groups[b]:
                    owners[k] = a
                groups[b].clear()
    return [sorted(i + 1 for i in g) for g in groups if g], overlapping


def make_split():
    groups, overlapping = scene_groups()
    label_hist = [np.bincount([grade for _, grade in load_regions(i)], minlength=3) for i in range(1, 61)]
    sizes = np.array([len(g) for g in groups])
    counts = np.array([sum((label_hist[i - 1] for i in group), np.zeros(3, dtype=int))
                       for group in groups])
    target = .18 * counts.sum(0)
    rng = np.random.default_rng(3122027)
    best = None
    best_loss = float("inf")
    for _ in range(12000):
        order = rng.permutation(len(groups))
        counts_so_far = np.cumsum(sizes[order])
        tests = order[counts_so_far <= 11]
        vals = order[(counts_so_far > 11) & (counts_so_far <= 22)]
        pair_counts = np.stack([counts[tests].sum(0), counts[vals].sum(0)])
        if pair_counts.min() < 20:
            continue
        loss = np.square((pair_counts - target) / target).sum()
        if loss < best_loss:
            best, best_loss = (tests, vals), loss
    if best is None:
        raise RuntimeError("Could not create train, validation and test partitions with all grades")
    test, val = [[scene for index in subset for scene in groups[index]] for subset in best]
    splits = {"train": sorted(set(range(1, 61)) - set(test) - set(val)),
              "val": sorted(val), "test": sorted(test)}
    report = {"splits": splits, "paired_scene_overlap": overlapping,
              "counts": {k: sum((label_hist[i - 1] for i in rows), np.zeros(3, int)).tolist()
                         for k, rows in splits.items()},
              "grade_order": ["A_no_visible_upper_damage", "B_partial", "C_collapsed"],
              "source": "https://doi.org/10.5281/zenodo.21296833"}
    write_json(DATA / "split.json", report)
    print(json.dumps(report, indent=2), flush=True)


def feature_record(num, degraded, size=112):
    pair = [read_rgb(DATA / kind / f"{num:02d}.jpg") for kind in ("pre", "post")]
    pre = pair[0]
    post = pair[1]
    if pre.shape != post.shape:
        raise ValueError(f"Scene {num} needs alignment metadata: {pre.shape}, {post.shape}")
    if degraded:
        ratio = .45
        images = [cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA) for img in pair]
    else:
        ratio, images = 1., pair
    pre, post = images
    feature_rows = []
    for bounds, class_id in load_regions(num):
        shifted = np.array(bounds) * ratio
        roi, mask = patch_pair(pre, post, shifted, size=size)
        feature_rows.append((roi, mask, signal(roi, mask), class_id))
    return feature_rows


def turkey_features(model, split, data, device, degraded):
    STORE.mkdir(parents=True, exist_ok=True)
    xs, fs, ys, ids = [], [], [], []
    scenes = data["splits"][split]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0, len(scenes), 3):
            group = scenes[offset:offset + 3]
            for num, rows in zip(group, pool.map(feature_record, group, [degraded] * len(group))):
                for start in range(0, len(rows), 72):
                    views, masks, stats, targets = zip(*rows[start:start + 72])
                    appearance, delta = embed(model, np.stack(views), np.stack(masks), device)
                    xs.append(appearance.astype(np.float16))
                    fs.append(np.concatenate((np.stack(stats), delta), axis=1))
                    ys.extend(targets)
                    ids.extend([num] * len(targets))
            if offset % 9 == 0 or offset + 3 >= len(scenes):
                print(f"Turkey {split}: scenes {min(offset + 3, len(scenes))}/{len(scenes)}, "
                      f"buildings {len(ys)}", flush=True)
    postfix = "_downsampled" if degraded else ""
    output = STORE / f"turkey_{split}{postfix}.npz"
    temporary = output.with_suffix(".tmp.npz")
    np.savez(temporary, appearance=np.concatenate(xs), change=np.concatenate(fs),
             grade=np.asarray(ys, np.uint8), chip=np.asarray(ids))
    temporary.replace(output)


def save_patches(split, data, size=112):
    scenes = data["splits"][split]
    count = int(sum(data["counts"][split]))
    store = DATA / ("patches" if size == 112 else f"patches{size}")
    store.mkdir(exist_ok=True)
    v = np.lib.format.open_memmap(store / f"{split}_rgb.npy", mode="w+", dtype=np.uint8,
                                  shape=(count, 2, size, size, 3))
    m = np.lib.format.open_memmap(store / f"{split}_roi.npy", mode="w+", dtype=np.uint8,
                                  shape=(count, size, size))
    metadata = []
    start = 0
    for scene in scenes:
        rows = feature_record(scene, False, size)
        end = start + len(rows)
        images, masks, signals, targets = zip(*rows)
        v[start:end] = np.stack(images)
        m[start:end] = np.stack(masks)
        metadata.extend(zip(signals, targets, [scene] * len(rows)))
        start = end
        print(f"Turkey patch cache {split}: {start}/{count}", flush=True)
    if start != count:
        raise ValueError(f"Expected {count} buildings; found {start}")
    v.flush()
    m.flush()
    x, y, ids = zip(*metadata)
    np.savez(store / f"{split}.npz", change=np.stack(x), grade=np.asarray(y), chip=np.asarray(ids))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inventory", action="store_true")
    cli.add_argument("--features", nargs="+", choices=("train", "val", "test"))
    cli.add_argument("--patches", nargs="+", choices=("train", "val", "test"))
    cli.add_argument("--size", type=int, default=112, choices=(112, 168))
    cli.add_argument("--degraded", action="store_true")
    args = cli.parse_args()
    cv2.setNumThreads(1)
    if args.inventory:
        unpack()
        make_split()
    if args.patches:
        splits = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
        for split in args.patches:
            save_patches(split, splits, args.size)
    if args.features:
        torch.set_num_threads(2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_dino(WEIGHTS, device)
        splits = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
        for split in args.features:
            turkey_features(model, split, splits, device, args.degraded)
