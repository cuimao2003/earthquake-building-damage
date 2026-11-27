"""Score an xBD folder's earthquake pairs with the locked larger classifier."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image
import torch

from infer_retrain import SelectedGrader, evaluate, load_localizers
from inventory import ROOT, raw_counts, scan, serial, write_json


def validate_images(row):
    fingerprints = []
    shapes = []
    root = Path(row["source"])
    for when in ("pre", "post"):
        name = f"{row['chip']}_{when}_disaster"
        file = root / "images" / f"{name}.png"
        with Image.open(file) as image:
            image.load()
            shapes.append(image.size)
        fingerprints.append(hashlib.sha256(file.read_bytes()).hexdigest())
    if shapes[0] != shapes[1]:
        raise ValueError(f"Mismatched image dimensions in {row['chip']}: {shapes}")
    label_file = root / "labels" / f"{row['chip']}_post_disaster.json"
    labels = json.loads(label_file.read_bytes())["features"]["xy"]
    target = sorted((item["properties"]["uid"], item["properties"].get("subtype"), item["wkt"])
                    for item in labels)
    return tuple(fingerprints), target


def collect(folder, out):
    seen = {}
    repeats = []
    source_counts = {}
    labels = sorted(folder.rglob("labels"), key=lambda p: (len(p.parts), str(p).casefold()))
    for directory in labels:
        if not directory.is_dir():
            continue
        rows = scan(directory.parent, "requested_test_folder")
        source_counts[str(directory)] = len(rows)
        for row in rows:
            checksums, target = validate_images(row)
            chip = row["chip"]
            if chip in seen:
                canonical, before, original_target = seen[chip]
                same_geometry = [(uid, box) for uid, _, box in target] == [
                    (uid, box) for uid, _, box in original_target]
                grades_omitted = same_geometry and all(grade is None for _, grade, _ in target)
                if before != checksums or (target != original_target and not grades_omitted):
                    raise ValueError(f"Conflicting versions of {chip}: {canonical['source']} and {row['source']}")
                repeats.append({"chip": chip, "source": row["source"], "kept": canonical["source"],
                                "damage_grades_omitted": grades_omitted})
            else:
                seen[chip] = row, checksums, target
    if not seen:
        raise ValueError(f"No valid earthquake image pairs under {folder}")

    training = json.loads((ROOT / "data/manifest.json").read_bytes())
    partition = {row["chip"]: name for name, rows in training.items() for row in rows}
    original_audit = json.loads((ROOT / "data/audit.json").read_bytes())
    excluded = {item["chip"]: item for item in original_audit["excluded"]}
    records = [serial(seen[chip][0]) for chip in sorted(seen)]
    history = {}
    for row in records:
        chip = row["chip"]
        history[chip] = partition.get(chip, "excluded_overlap" if chip in excluded else "unlisted")
    audit = {"requested_folder": str(folder), "source_counts": source_counts,
             "unique_pairs": len(records), "duplicate_pairs": repeats,
             "prior_partitions": dict(Counter(history.values())), "prior_partition_by_chip": history,
             "previously_excluded": [excluded[r["chip"]] for r in records if r["chip"] in excluded],
             "annotation_counts": raw_counts(records)}
    write_json(out / "input_audit.json", audit)
    write_json(out / "input_manifest.json", records)
    print(json.dumps({k: audit[k] for k in ("unique_pairs", "prior_partitions", "annotation_counts")}), flush=True)
    return records, history


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("folder", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--classifier", type=Path, default=ROOT / "run/mexico_large_selection")
    cli.add_argument("--previous-validation-only", action="store_true",
                     help="Score only pairs previously assigned to model validation")
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    rows, history = collect(args.folder.resolve(), args.out)
    if args.previous_validation_only:
        rows = [row for row in rows if history[row["chip"]] == "val"]
        if not rows:
            cli.error("No earthquake pairs from previous validation were found in this folder")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grader = SelectedGrader(args.classifier, device)
    mode = grader.calibrated_localizer
    if not mode:
        cli.error("A locked automatic calibration is required")
    write_json(args.out / "model.json", {"classifier": str(args.classifier.resolve()), "localizer": mode,
                                        "evaluated_pairs": len(rows)})
    evaluate(rows, grader, mode, load_localizers(device), args.out, known_boxes=True, pixel_audit=True)
