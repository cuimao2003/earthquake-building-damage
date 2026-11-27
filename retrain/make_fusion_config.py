"""Build a ProposalGrader-compatible fusion.json from a run_fusion_grid.py output."""

import argparse
import hashlib
import json
from pathlib import Path

import train_proposal_transformer as T


def digest(path):
    fingerprint = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    return fingerprint.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    grid = json.loads(args.grid.read_bytes())
    selected = dict(grid["selected"])
    selected["validity_threshold"] = 0.0
    blend = {
        "candidate_sha256": digest(args.head),
        "reference": str(T.REFERENCE.resolve()),
        "reference_files": T.reference_files(T.REFERENCE),
        "scenes": grid["scenes"],
        "source": str(args.grid.resolve()),
        "selected": selected,
        "trials": grid["trials"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blend, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = selected["validation"]
    print(f"wrote {args.out} | weight {selected['weight']} | macro {validation['macro_f1']:.4f} "
          f"| damaged {validation['damaged_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
