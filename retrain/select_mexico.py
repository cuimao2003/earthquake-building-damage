"""Lock a Mexican damage classifier or grade-specific ensemble on validation."""

import argparse
import json
from pathlib import Path

import numpy as np

from classifier import calibrate, load_data, odds, RUN
from inventory import write_json


DEFAULT_CANDIDATES = ("mexico", "mexico_transfer", "mexico_joint", "mexico_herat", "mexico_herat_only")


def likelihood_pair(light, severe):
    margins = np.column_stack((odds(light)[:, 0], odds(severe)[:, 1]))
    logits = np.column_stack((np.zeros(len(margins)), margins))
    values = np.exp(logits - logits.max(1, keepdims=True))
    return values / values.sum(1, keepdims=True)


def head_scores(path):
    setup = json.loads((path / "selection.json").read_bytes())
    weight = setup["selected"]["texture_weight"]
    head = np.load(path / "val_head_scores.npy")
    if weight:
        tree = np.load(path / "val_texture_scores.npy")
        head = (1 - weight) * head + weight * tree
    encoder = (Path(setup.get("encoder_checkpoint", RUN / "turkey_vit/turkey_vit.pt"))
               if setup.get("transferred_encoder") else None)
    return {"model": str(path.resolve()), "encoder": str(encoder.resolve()) if encoder else None}, head


def run_selection(names=DEFAULT_CANDIDATES, out=RUN / "mexico_final"):
    targets = load_data("val")["grade"]
    candidates = [head_scores(RUN / name) for name in names
                  if (RUN / name / "selection.json").exists()]
    if len(candidates) < 2:
        raise ValueError("Expected at least two independently trained classifiers")
    results = []
    for light, first in candidates:
        for severe, second in candidates:
            scores = likelihood_pair(first, second)
            limits, metrics = calibrate(scores, targets)
            results.append({"minor": light, "major": severe, "thresholds": limits, "validation": metrics})
    best = max(results, key=lambda option: option["validation"]["macro_f1"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "selection.json", {"selected": best, "candidates": results})
    print(json.dumps(best, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=RUN / "mexico_final")
    cli.add_argument("--add", action="append", default=[], help="Add a classifier run under run/")
    args = cli.parse_args()
    missing = [name for name in args.add if not (RUN / name / "selection.json").is_file()]
    if missing:
        cli.error(f"Incomplete classifier runs: {', '.join(missing)}")
    run_selection((*DEFAULT_CANDIDATES, *args.add), args.out)
