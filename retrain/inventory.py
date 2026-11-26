"""Lock an unseen holdout before retraining and audit scene overlap."""

import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from shapely import wkt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from prepare import overlap, tile_extent, write_json

LABELS = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 2}
INPUT = Path("D:/pc/earthquake data")
SOURCES = {"train_original": INPUT / "train", "test_previous": INPUT / "test/test",
           "hold_new": ROOT / "data/original/hold", "tier3": ROOT / "data/original/tier3"}


def scan(source, origin):
    results = []
    for path in sorted((source / "labels").glob("*earthquake*_post_disaster.json")):
        chip = path.name[:-len("_post_disaster.json")]
        target = json.loads(path.read_bytes())
        if target["metadata"]["disaster_type"] != "earthquake":
            continue
        images = [source / "images" / f"{chip}_{date}_disaster.png" for date in ("pre", "post")]
        if not all(path.is_file() and path.stat().st_size > 0 for path in images):
            raise ValueError(f"Incomplete images for {chip}")
        annotated = {}
        labels = Counter()
        for element in target["features"]["xy"]:
            props = element["properties"]
            annotated[props["uid"]] = wkt.loads(element["wkt"])
            labels[props.get("subtype", "unknown")] += 1
        metadata = target["metadata"]
        bounds = tile_extent(annotated, target["features"]["lng_lat"], metadata["width"], metadata["height"])
        results.append({"chip": chip, "source": str(source.resolve()), "origin": origin,
                        "counts": dict(labels), "captures": metadata["catalog_id"],
                        "extent": bounds, "uids": set(annotated)})
    return results


def group_tiles(records):
    groups = [{i} for i in range(len(records))]
    owners = list(range(len(records)))
    for i, tile in enumerate(records):
        for j, other in enumerate(records[:i]):
            a, b = owners[i], owners[j]
            if a == b:
                continue
            if tile["uids"] & other["uids"] or overlap(tile["extent"], other["extent"]) > 0.003:
                groups[a].update(groups[b])
                for member in groups[b]:
                    owners[member] = a
                groups[b].clear()
    return [sorted(group) for group in groups if group]


def raw_counts(records):
    counts = Counter()
    for row in records:
        counts.update(row["counts"])
    return dict(counts)


def val_choice(records):
    groups = group_tiles(records)
    sizes = np.asarray([len(g) for g in groups])
    grouped = [raw_counts([records[i] for i in g]) for g in groups]
    counts = np.array([[c.get("minor-damage", 0), c.get("major-damage", 0) + c.get("destroyed", 0)]
                       for c in grouped])
    # Keep small clusters intact and choose a validation partition with
    # positives in both grades, reflecting the natural class prevalence.
    n_tiles = max(1, round(len(records) * 0.17))
    target = np.maximum(1, np.round(counts.sum(0) * 0.19))
    rng = np.random.default_rng(780172)
    best, error = None, np.inf
    for _ in range(15000):
        indices = rng.permutation(len(groups))
        take = indices[np.cumsum(sizes[indices]) <= n_tiles]
        positives = counts[take].sum(0)
        if not take.size or min(positives) == 0 or min(counts.sum(0) - positives) == 0:
            continue
        score = np.square((positives - target) / np.sqrt(target)).sum()
        if score < error:
            best, error = take, score
    if best is None:
        raise ValueError("Could not reserve validation tiles with both damage grades")
    return {i for g in best for i in groups[g]}


def serial(record):
    return {name: value for name, value in record.items() if name not in ("extent", "uids")}


def prepare():
    sources = {origin: scan(source, origin) for origin, source in SOURCES.items()}
    hold = sources["hold_new"]
    if len(hold) != 34:
        raise ValueError(f"Expected the 34 newly recovered hold tiles, got {len(hold)}")
    dev, excluded, seen = [], [], {r["chip"] for r in hold}
    for origin in ("train_original", "test_previous", "tier3"):
        for row in sources[origin]:
            max_overlap, closest = max((overlap(row["extent"], r["extent"]), r["chip"]) for r in hold)
            shared = sum(bool(row["uids"] & r["uids"]) for r in hold)
            if row["chip"] in seen or shared or max_overlap > 0.003:
                excluded.append({"chip": row["chip"], "origin": origin, "hold_chip": closest,
                                 "overlap": max_overlap, "matches_id": row["chip"] in seen,
                                 "shared_building_ids": shared})
                continue
            seen.add(row["chip"])
            dev.append(row)
    val = val_choice(dev)
    splits = {"train": [r for i, r in enumerate(dev) if i not in val],
              "val": [r for i, r in enumerate(dev) if i in val], "test": hold}
    summary = {name: {"tiles": len(rows), "counts": raw_counts(rows)} for name, rows in splits.items()}
    audit = {"summary": summary, "excluded": excluded,
             "unavailable_geography": [r["chip"] for rows in sources.values() for r in rows
                                       if r["extent"] is None]}
    write_json(ROOT / "data/manifest.json", {k: [serial(r) for r in v] for k, v in splits.items()})
    write_json(ROOT / "data/audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
