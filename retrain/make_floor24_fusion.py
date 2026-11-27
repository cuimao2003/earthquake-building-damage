"""Build a ProposalGrader-compatible fusion.json from run_fusion_grid.py output."""

import hashlib
import json
from pathlib import Path

import train_proposal_transformer as T

BASE = Path(__file__).resolve().parent
HEAD = BASE / "run/proposal_attention_floor24/head.pt"
GRID = BASE / "run/fusion_grid_floor24.json"
OUT = BASE / "run/floor24_fusion/fusion.json"


def digest(path):
    fingerprint = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    return fingerprint.hexdigest()


def main():
    grid = json.loads(GRID.read_bytes())
    selected = dict(grid["selected"])
    selected["validity_threshold"] = 0.0
    blend = {
        "candidate_sha256": digest(HEAD),
        "reference": str(T.REFERENCE.resolve()),
        "reference_files": T.reference_files(T.REFERENCE),
        "scenes": grid["scenes"],
        "source": "run_fusion_grid.py on data/proposal_lower_floor (detector floor 0.24)",
        "selected": selected,
        "trials": grid["trials"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blend, ensure_ascii=False, indent=2), encoding="utf-8")
    val = selected["validation"]
    print(f"wrote {OUT}")
    print(f"weight {selected['weight']}, macro F1 {val['macro_f1']:.4f}, "
          f"damaged F1 {val['damaged_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
