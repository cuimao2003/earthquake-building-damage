"""Cross-event evaluation of the locked classifiers on an external xBD-format folder.

Unlike evaluate_test_folder.collect this accepts any disaster name (for example the
Palu 2018 earthquake + tsunami test split), so it can be used as an independent
cross-earthquake test set.

Usage:
  python evaluate_cross_event.py --folder data/external/palu_test --disaster palu-tsunami \
      --mode old --out run/cross_event_palu_old --result run/cross_event_palu_old
  python evaluate_cross_event.py --folder data/external/palu_test --disaster palu-tsunami \
      --mode proposal --out run/proposal_attention_floor24 \
      --fusion run/floor24_fusion/fusion.json \
      --weights weights/dinov2_vitb14_pretrain.pth --result run/cross_event_palu_floor24_fusion
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

import train_proposal_transformer as T
from infer_retrain import SelectedGrader, evaluate, load_localizers
from inventory import ROOT, write_json
from train_proposal_transformer import NAME


def records_for(folder, disaster):
    records = []
    for path in sorted(folder.joinpath("labels").glob("*_post_disaster.json")):
        chip = path.name[: -len("_post_disaster.json")]
        target = json.loads(path.read_bytes())
        metadata = target["metadata"]
        if metadata.get("disaster") != disaster:
            continue
        for when in ("pre", "post"):
            image = folder / "images" / f"{chip}_{when}_disaster.png"
            if not image.is_file() or image.stat().st_size == 0:
                raise ValueError(f"Incomplete images for {chip}")
        counts = Counter(item["properties"].get("subtype", "unknown") for item in target["features"]["xy"])
        records.append({"source": str(folder.resolve()), "chip": chip, "counts": dict(counts)})
    if not records:
        raise ValueError(f"No pairs for disaster {disaster} under {folder}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--disaster", required=True)
    parser.add_argument("--mode", choices=("old", "proposal"), default="old")
    parser.add_argument("--out", type=Path, required=True,
                        help="proposal mode: trained head run directory; old mode: unused")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--fusion", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/dinov2_vitb14_pretrain.pth")
    parser.add_argument("--detector", type=Path, default=ROOT / "run/swin_loc_mix/detector.pt")
    parser.add_argument("--classification", type=Path, default=ROOT / "run/mexico_large_selection")
    parser.add_argument("--encode-batch", type=int, default=8)
    parser.add_argument("--reference", type=Path, default=ROOT / "run/mexico_large_selection")
    parser.add_argument("--manifest", type=Path, default=T.MANIFEST,
                        help="manifest used to train the proposal head (default: the Mexico-only manifest)")
    args = parser.parse_args()

    if args.result.exists() and any(args.result.iterdir()):
        raise ValueError("Use a fresh --result directory")
    args.result.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = records_for(args.folder.resolve(), args.disaster)
    counts = Counter()
    for row in records:
        counts.update(row["counts"])
    print(f"{args.disaster}: {len(records)} pairs, {dict(counts)}", flush=True)
    write_json(args.result / "protocol.json", {"folder": str(args.folder.resolve()), "disaster": args.disaster,
                                               "mode": args.mode, "pairs": len(records),
                                               "annotation_counts": dict(counts)})

    if args.mode == "old":
        grader = SelectedGrader(args.classification, device)
        mode = grader.calibrated_localizer
        if not mode:
            raise ValueError("A locked automatic calibration is required")
        write_json(args.result / "model.json", {"classifier": str(args.classification.resolve()),
                                                "localizer": mode, "evaluated_pairs": len(records)})
        evaluate(records, grader, mode, load_localizers(device), args.result, known_boxes=True, pixel_audit=False)
    else:
        checkpoint = torch.load(args.out / "head.pt", map_location="cpu")
        if checkpoint["cache"]["manifest_sha256"] != T.digest(args.manifest):
            raise ValueError("Evaluation requires the same manifest that was used for training")
        wrapper = SimpleNamespace(out=args.out, fusion=args.fusion, weights=args.weights,
                                  detector=args.detector, encode_batch=args.encode_batch)
        grader = T.ProposalGrader(wrapper, device)
        write_json(args.result / "model.json", {"head": str((args.out / "head.pt").resolve()),
                                                "fusion": str(args.fusion.resolve()) if args.fusion else None,
                                                "evaluated_pairs": len(records)})
        evaluate(records, grader, NAME + "_fusion" if args.fusion else NAME, None, args.result,
                 known_boxes=True, pixel_audit=False, boxes_from=grader.propose,
                 registration_method=grader.registration)


if __name__ == "__main__":
    main()
