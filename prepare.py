"""Validate the local imagery and prepare memory-mapped training data."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon


BUILDING_ROOT = Path("F:/transformer/building train")
EARTHQUAKE_ROOT = Path("D:/pc/earthquake data")
DATA = Path(__file__).resolve().parent / "cache"
SIZE = 256
CLASSES = ("background", "no-damage", "minor-damage", "major-or-destroyed")
RAW_LEVEL = {"no-damage": 1, "minor-damage": 2, "major-damage": 3,
             "destroyed": 3, "un-classified": 255}


def write_json(path, content):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def decode(path, flags):
    raw = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image, hashlib.sha256(raw).hexdigest()


def building_chip(item):
    image_path, label_path = item
    try:
        # OpenCV avoids a TIFF codec failure in the existing Conda environment.
        rgb, _ = decode(image_path, cv2.IMREAD_COLOR)
        label, _ = decode(label_path, cv2.IMREAD_UNCHANGED)
        if label.ndim == 3:
            if not all(np.array_equal(channel, label[:, :, 0]) for channel in
                       np.moveaxis(label[:, :, :3], 2, 0)):
                raise ValueError("colored building mask")
            label = label[:, :, 0]
        if rgb.shape[:2] != label.shape:
            raise ValueError("image/mask size mismatch")
        if not np.isin(np.unique(label), (0, 1, 255)).all():
            raise ValueError(f"unexpected label values {np.unique(label).tolist()}")
        rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        return (image_path.name, rgb[:, :, ::-1], label > 0), None
    except (OSError, ValueError, cv2.error) as error:
        return None, [image_path.name, str(error)]


def prepare_buildings(root, cache, limit):
    rng = np.random.default_rng(5482)
    names = sorted((root / "image").glob("*.tif"))
    masks = {path.name: path for path in (root / "label").glob("*.tif")}
    candidates = defaultdict(list)
    for path in names:
        if path.name in masks:
            source, _, index = path.stem.rpartition("_")
            candidates[(source, int(index) // 64 if index.isdigit() else index)].append(
                (path, masks[path.name]))
    if not candidates:
        raise ValueError(f"No matching building TIFFs in {root}")
    blocks = sorted(candidates)
    holdout = {blocks[i] for i in rng.permutation(len(blocks))[:max(1, len(blocks) // 10)]}
    result = {"source": str(root), "matched": sum(map(len, candidates.values())),
              "images_on_disk": len(names), "labels_on_disk": len(masks)}
    for split, budget in (("train", limit), ("val", max(1, limit // 10))):
        pool = [item for key in blocks if (key in holdout) == (split == "val")
                for item in candidates[key]]
        order = [pool[i] for i in rng.permutation(len(pool))]
        budget = min(budget, len(pool))
        array = np.lib.format.open_memmap(cache / f"building_{split}.npy", mode="w+",
                                           dtype=np.uint8, shape=(budget, SIZE, SIZE, 4))
        accepted, rejected = [], []
        with ThreadPoolExecutor(max_workers=4) as workers:
            # Keep a small bounded queue of decoded images in host memory.
            for offset in range(0, len(order), 32):
                if len(accepted) == budget:
                    break
                for sample, error in workers.map(building_chip, order[offset:offset + 32]):
                    if error:
                        rejected.append(error)
                        continue
                    name, rgb, label = sample
                    array[len(accepted), :, :, :3] = rgb
                    array[len(accepted), :, :, 3] = label
                    accepted.append(name)
                    if len(accepted) == budget:
                        break
                if len(accepted) and len(accepted) // 512 != (len(accepted) - 32) // 512:
                    print(f"building {split}: {len(accepted)}/{budget}", flush=True)
        array.flush()
        if not accepted:
            raise RuntimeError(f"No readable building {split} examples")
        result[split] = {"names": accepted, "rejected": rejected}
    write_json(cache / "building.json", result)


def read_label(path):
    doc = json.loads(path.read_bytes())
    md = doc["metadata"]
    if md["disaster_type"] != "earthquake" or md["img_name"] != path.stem + ".png":
        raise ValueError(f"Incorrect label metadata in {path}")
    return doc["features"]


def paint(mask, geometry, value):
    polygons = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
    for polygon in polygons:
        if polygon.is_empty:
            continue
        rings = [polygon.exterior, *polygon.interiors]
        vertices = [np.round(np.asarray(ring.coords) * 16).astype(np.int32) for ring in rings]
        cv2.fillPoly(mask, vertices, value, shift=4)


def tile_extent(xy, geographic, width, height):
    points = {r["properties"]["uid"]: wkt.loads(r["wkt"]).centroid for r in geographic}
    pairs = [(shape.centroid.coords[0], points[uid].coords[0]) for uid, shape in xy.items()
             if uid in points and not shape.is_empty and not points[uid].is_empty]
    if len(pairs) < 4:
        return None
    pixels, lnglat = np.asarray(pairs).transpose(1, 0, 2)
    design = np.c_[pixels, np.ones(len(pixels))]
    fit, _, rank, _ = np.linalg.lstsq(design, lnglat, rcond=None)
    extent = Polygon(np.c_[[(0, 0), (width, 0), (width, height), (0, height)], np.ones(4)] @ fit)
    return extent if rank == 3 and extent.is_valid and extent.area > 0 else None


def stage_tile(root, chip, staging):
    images, hashes, labels, original_mask_counts = [], [], [], []
    for when in ("pre", "post"):
        stem = f"{chip}_{when}_disaster"
        image, digest = decode(root / "images" / (stem + ".png"), cv2.IMREAD_COLOR)
        images.append(image[:, :, ::-1])
        hashes.append(digest)
        labels.append(read_label(root / "labels" / (stem + ".json")))
        target_path = root / "targets" / (stem + "_target.png")
        target, _ = decode(target_path, cv2.IMREAD_UNCHANGED)
        if target.shape != image.shape[:2] or not np.isin(np.unique(target), (0, 1, 2, 3, 4, 5)).all():
            raise ValueError(f"Invalid xBD mask {target_path}")
        original_mask_counts.append(np.bincount(target.ravel(), minlength=6).tolist())
    height, width = images[1].shape[:2]
    if images[0].shape != images[1].shape or height < SIZE or width < SIZE:
        raise ValueError(f"Invalid pair dimensions {chip}")
    post = labels[1]
    ordered = sorted(post["xy"], key=lambda r: RAW_LEVEL[r["properties"]["subtype"]])
    damage = np.zeros((height, width), np.uint8)
    buildings, xy, counts = [], {}, Counter()
    for entry in ordered:
        uid = entry["properties"]["uid"]
        level = entry["properties"]["subtype"]
        shape = wkt.loads(entry["wkt"])
        if shape.is_empty or shape.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Invalid building geometry {chip}/{uid}")
        if uid in xy:
            raise ValueError(f"Duplicate building ID {chip}/{uid}")
        paint(damage, shape, RAW_LEVEL[level])
        xy[uid] = shape
        counts[level] += 1
        center = list(shape.representative_point().coords)[0]
        buildings.append({"uid": uid, "label": level, "class_id": RAW_LEVEL[level],
                          "center": center, "bounds": list(shape.bounds)})
    path = staging / f"{chip}.npy"
    np.save(path, np.dstack([*images, damage]))
    extent = tile_extent(xy, post["lng_lat"], width, height)
    return {"chip": chip, "file": path.name, "source": str(root), "hashes": hashes,
            "shape": [height, width], "building_counts": dict(counts), "buildings": buildings,
            "original_mask_counts": original_mask_counts,
            "pixels": np.bincount(damage.ravel(), minlength=256)[[0, 1, 2, 3, 255]].tolist(),
            "extent": list(extent.exterior.coords) if extent is not None else None}


def overlap(a, b):
    if a is None or b is None or not a.intersects(b):
        return 0.
    return a.intersection(b).area / min(a.area, b.area)


def summarize(records):
    levels = Counter()
    pixels = np.zeros(5, np.int64)
    for record in records:
        levels.update(record["building_counts"])
        pixels += record["pixels"]
    return {"pairs": len(records), "buildings": dict(levels), "pixels": pixels.tolist()}


def select_validation(records, related):
    groups = []
    pending = set(range(len(records)))
    while pending:
        component = {min(pending)}
        last = set()
        while component != last:
            last = set(component)
            component.update(j for i in last for j in related[i] & pending)
        pending -= component
        groups.append(sorted(component))
    counts = np.array([[sum(records[i]["building_counts"].get(key, 0) for i in group)
                        for key in ("minor-damage", "major-damage", "destroyed")]
                       for group in groups])
    sizes = np.array([len(g) for g in groups])
    # Reserve buildings at both learnable damage grades for validation and keep
    # the two available fully destroyed examples for training.
    choices = np.flatnonzero(counts[:, 2] == 0)
    target = np.array([19, 5, 0])
    rng = np.random.default_rng(4307)
    best, best_score = None, float("inf")
    for _ in range(12000):
        order = rng.permutation(choices)
        picked = order[np.cumsum(sizes[order]) <= max(1, round(0.16 * len(records)))]
        observed = counts[picked].sum(axis=0)
        remaining = counts.sum(axis=0) - observed
        if not len(picked) or min(observed[:2]) == 0 or min(remaining[:2]) == 0:
            continue
        score = ((observed[0] - target[0]) / 6) ** 2 + (observed[1] - target[1]) ** 2
        if score < best_score:
            best, best_score = picked, score
        if best_score == 0:
            break
    if best is None:
        raise ValueError("Cannot create validation with both damage grades represented")
    return {i for g in best for i in groups[g]}


def prepare_earthquake(root, cache):
    staging = cache / "earthquake"
    staging.mkdir(exist_ok=True)
    sources = [root / "train", root / "test"]
    if not (sources[1] / "images").is_dir():
        sources[1] /= "test"
    raw_sets, rejected = [], []
    seen = set()
    for source in sources:
        chips = sorted({p.name.rsplit("_", 2)[0] for p in (source / "images").glob("*earthquake*.png")})
        records = []
        for count, chip in enumerate(chips, 1):
            try:
                if chip in seen:
                    raise ValueError(f"duplicate split ID {chip}")
                record = stage_tile(source, chip, staging)
                records.append(record)
                seen.add(chip)
            except (OSError, ValueError, KeyError, cv2.error) as error:
                rejected.append({"source": str(source), "chip": chip, "reason": str(error)})
            if count % 20 == 0 or count == len(chips):
                print(f"{source.name}: {count}/{len(chips)} checked", flush=True)
        if not records:
            raise RuntimeError(f"No usable earthquake pairs in {source}")
        raw_sets.append(records)
    primary, test = raw_sets
    extents = {r["chip"]: Polygon(r["extent"]) if r["extent"] else None
               for r in primary + test}
    test_ids = {b["uid"] for r in test for b in r["buildings"]}
    test_hashes = {h for r in test for h in r["hashes"]}
    retained, excluded = [], []
    for record in primary:
        maximum = max(((overlap(extents[record["chip"]], extents[q["chip"]]), q["chip"])
                       for q in test))
        same = bool(test_hashes.intersection(record["hashes"]))
        same_uid = any(b["uid"] in test_ids for b in record["buildings"])
        if same or same_uid or maximum[0] > 0.02:
            excluded.append({"chip": record["chip"], "test_overlap": maximum[0],
                             "test_chip": maximum[1], "same_image": same, "same_uid": same_uid})
        else:
            retained.append(record)
    related = [set() for _ in retained]
    uid_sets = [{b["uid"] for b in r["buildings"]} for r in retained]
    for i, record in enumerate(retained):
        for j in range(i):
            other = retained[j]
            if (set(record["hashes"]).intersection(other["hashes"])
                    or uid_sets[i].intersection(uid_sets[j])
                    or overlap(extents[record["chip"]], extents[other["chip"]]) > 0.02):
                related[i].add(j)
                related[j].add(i)
    val_ids = select_validation(retained, related)
    splits = {"train": [r for i, r in enumerate(retained) if i not in val_ids],
              "val": [r for i, r in enumerate(retained) if i in val_ids], "test": test}
    summary = {key: summarize(rows) for key, rows in splits.items()}
    audit = {"classes": CLASSES, "ignored": 255, "splits": summary,
             "excluded_overlap": excluded, "rejected": rejected,
             "geographic_extent_unavailable": [k for k, v in extents.items() if v is None],
             "raw_sources": [summarize(records) for records in raw_sets]}
    write_json(cache / "earthquake.json", splits)
    write_json(cache / "audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["building", "earthquake"])
    parser.add_argument("--building-root", type=Path, default=BUILDING_ROOT)
    parser.add_argument("--earthquake-root", type=Path, default=EARTHQUAKE_ROOT)
    parser.add_argument("--building-limit", type=int, default=4096)
    parser.add_argument("--cache", type=Path, default=DATA)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    if args.source == "building":
        prepare_buildings(args.building_root, args.cache, args.building_limit)
    else:
        prepare_earthquake(args.earthquake_root, args.cache)


if __name__ == "__main__":
    main()
