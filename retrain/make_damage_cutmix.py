"""Build damage copy-paste (CutMix-style) synthetic training scenes for grading.

For each target xBD training scene, replace a handful of intact buildings with damaged
buildings copied from other scenes: the donor crop is pasted into both the pre and post
images at the target building location, and the post label is rewritten to the donor
damage grade. This creates many more light/heavy damaged examples while keeping the
pre/post difference realistic.

Usage:
  python make_damage_cutmix.py --out data/synthetic_cutmix --manifest data/manifest_cutmix.json \
      --scenes 130 --per-scene 8
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from inventory import ROOT
from shapely import wkt

LABELS = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 2}
DAMAGE_NAMES = ("minor-damage", "major-damage", "destroyed")


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_labels(path):
    return json.loads(Path(path).read_bytes())


def rect_wkt(box):
    x1, y1, x2, y2 = box
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    return "POLYGON ((" + ", ".join(f"{x:.1f} {y:.1f}" for x, y in points) + "))"


def scene_paths(row):
    root = Path(row["source"])
    return (root / "images" / f"{row['chip']}_pre_disaster.png",
            root / "images" / f"{row['chip']}_post_disaster.png",
            root / "labels" / f"{row['chip']}_post_disaster.json")


def load_scene(row):
    pre_path, post_path, label_path = scene_paths(row)
    pre = read_rgb(pre_path)
    post = read_rgb(post_path)
    labels = read_labels(label_path)
    boxes, grades = [], []
    for item in labels["features"]["xy"]:
        bounds = wkt.loads(item["wkt"]).bounds
        boxes.append(np.asarray(bounds, np.float32))
        grades.append(LABELS.get(item["properties"].get("subtype"), -1))
    return pre, post, labels, np.asarray(boxes, np.float32), np.asarray(grades, np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest_plus_palu.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/synthetic_cutmix")
    parser.add_argument("--output-manifest", type=Path, default=ROOT / "data/manifest_cutmix.json")
    parser.add_argument("--scenes", type=int, default=130)
    parser.add_argument("--per-scene", type=int, default=8)
    parser.add_argument("--donors", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    manifest = json.loads(args.manifest.read_bytes())
    train_rows = manifest["train"]

    # Collect donor damaged-building crops (preloaded to keep the generation loop fast).
    donor_pool = []
    for row in train_rows:
        pre, post, labels, boxes, grades = load_scene(row)
        for index, item in enumerate(labels["features"]["xy"]):
            grade = int(grades[index])
            if grade <= 0:
                continue
            box = boxes[index]
            x1, y1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
            x2 = min(pre.shape[1], int(round(box[2])))
            y2 = min(pre.shape[0], int(round(box[3])))
            if x2 - x1 < 18 or y2 - y1 < 18:
                continue
            crop_shape = (y2 - y1, x2 - x1)
            polygon = np.asarray(wkt.loads(item["wkt"]).exterior.coords, np.float32) - np.array([x1, y1], np.float32)
            mask = np.zeros(crop_shape, np.uint8)
            cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
            if mask.sum() < 40:
                continue
            donor_pool.append({"grade": grade, "subtype": item["properties"]["subtype"],
                               "pre": pre[y1:y2, x1:x2].copy(), "post": post[y1:y2, x1:x2].copy(),
                               "mask": mask})
    print(f"donor damaged buildings: {len(donor_pool)}", flush=True)
    if not donor_pool:
        raise SystemExit("no damaged donors found")
    minor = [d for d in donor_pool if d["grade"] == 1]
    major = [d for d in donor_pool if d["grade"] == 2]
    rng.shuffle(minor)
    rng.shuffle(major)
    # balance light/heavy donors (hardest class gets equal representation)
    half = args.donors // 2
    keep_minor = [minor[i % len(minor)] for i in range(half)] if minor else []
    keep_major = major[: args.donors - len(keep_minor)]
    donors = keep_minor + keep_major
    rng.shuffle(donors)
    print(f"selected donors: {len(donors)} (minor {len(keep_minor)}, major/destroyed {len(keep_major)})",
          flush=True)

    targets = [row for row in train_rows if row["counts"].get("no-damage", 0) >= 12]
    rng.shuffle(targets)
    targets = targets[: args.scenes]
    out_images = args.out / "images"
    out_labels = args.out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    synthetic_rows = []
    for scene_index, row in enumerate(targets, 1):
        pre, post, labels, boxes, grades = load_scene(row)
        intact = [index for index, grade in enumerate(grades) if grade == 0
                  and 14 <= boxes[index][2] - boxes[index][0] <= 130
                  and 14 <= boxes[index][3] - boxes[index][1] <= 130]
        rng.shuffle(intact)
        sites = []
        for index in intact:
            center = np.array([(boxes[index][0] + boxes[index][2]) / 2,
                               (boxes[index][1] + boxes[index][3]) / 2])
            if any(np.linalg.norm(center - other) < 22 for other in sites):
                continue
            sites.append(center)
            if len(sites) >= args.per_scene:
                break
        pasted = 0
        for center in sites:
            distances = [np.linalg.norm(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) - center) for b in boxes]
            index = int(np.argmin(distances))
            if grades[index] != 0:
                continue
            box = boxes[index]
            x1, y1 = int(round(box[0])), int(round(box[1]))
            x2, y2 = int(round(box[2])), int(round(box[3]))
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            donor = donors[int(rng.integers(len(donors)))]
            resized_pre = cv2.resize(donor["pre"], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            resized_post = cv2.resize(donor["post"], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(donor["mask"], (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 0
            target_pre = pre[y1:y2, x1:x2]
            target_post = post[y1:y2, x1:x2]
            target_pre[resized_mask] = resized_pre[resized_mask]
            target_post[resized_mask] = resized_post[resized_mask]
            pre[y1:y2, x1:x2] = target_pre
            post[y1:y2, x1:x2] = target_post
            label = labels["features"]["xy"][index]
            label["properties"]["subtype"] = donor["subtype"]
            label["properties"]["uid"] = f"cutmix-{row['chip']}-{index}"
            label["wkt"] = rect_wkt((x1, y1, x2, y2))
            grades[index] = donor["grade"]
            pasted += 1
        if pasted == 0:
            continue
        chip = f"cutmix-{row['chip']}"
        cv2.imwrite(str(out_images / f"{chip}_pre_disaster.png"), cv2.cvtColor(pre, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_images / f"{chip}_post_disaster.png"), cv2.cvtColor(post, cv2.COLOR_RGB2BGR))
        (out_labels / f"{chip}_post_disaster.json").write_bytes(
            json.dumps(labels, ensure_ascii=False).encode("utf-8"))
        counts = {}
        for item in labels["features"]["xy"]:
            subtype = item["properties"].get("subtype", "un-classified")
            counts[subtype] = counts.get(subtype, 0) + 1
        synthetic_rows.append({"chip": chip, "source": str(args.out), "origin": "cutmix",
                               "counts": counts, "captures": row.get("captures", [])})
        if scene_index % 20 == 0:
            print(f"scene {scene_index}/{len(targets)}: synthetic {len(synthetic_rows)}", flush=True)
    output = {"train": train_rows + synthetic_rows, "val": manifest["val"], "test": manifest["test"]}
    args.output_manifest.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = {}
    for row in synthetic_rows:
        for name, count in row["counts"].items():
            stats[name] = stats.get(name, 0) + count
    print(f"synthetic scenes: {len(synthetic_rows)} | damage counts: {json.dumps(stats, ensure_ascii=False)}")
    print(f"train total: {len(train_rows) + len(synthetic_rows)} | manifest {args.output_manifest}")


if __name__ == "__main__":
    main()
