"""Hybrid evaluation: TTA+WBF localization proposals graded by the locked old classifier.

This keeps the old pipeline's stronger given-box grading (ViT-L/ViT-B + texture) but
replaces its localizer with flip-TTA merged Swin proposals. Optionally filters the
proposals with the validity branch of a proposal-Transformer head.

Usage:
  python evaluate_hybrid_tta_old.py --split val --tta h --threshold 0.36 \
      --result run/hybrid_tta_old_val
  python evaluate_hybrid_tta_old.py --split val --tta h --threshold 0.36 \
      --validity-head run/p1_tta_pooled --validity-threshold 0.65 --result run/hybrid_tta_old_val_filtered
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_proposal_transformer as T
from infer_retrain import SelectedGrader, evaluate
from inventory import ROOT
from roi_transformer import RoofTransformer, encode_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--detector", type=Path, default=ROOT / "run/swin_loc_mix/detector.pt")
    parser.add_argument("--classifier", type=Path, default=ROOT / "run/mexico_large_selection")
    parser.add_argument("--tta", choices=("none", "h", "hv"), default="h")
    parser.add_argument("--threshold", type=float, default=0.36)
    parser.add_argument("--validity-head", type=Path, default=None)
    parser.add_argument("--validity-threshold", type=float, default=0.0)
    parser.add_argument("--encode-batch", type=int, default=24)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, detector_state = T.load_swin(args.detector, device)
    grader = SelectedGrader(args.classifier, device)
    encoder = head = None
    if args.validity_head is not None:
        encoder = T.load_encoder(ROOT / "weights/dinov2_vitb14_pretrain.pth", device)
        checkpoint = torch.load(args.validity_head / "head.pt", map_location="cpu")
        head = RoofTransformer(checkpoint["width"], checkpoint["variant"]).to(device)
        head.load_state_dict(checkpoint["head"])
        head.eval()

    def proposals(pre, post):
        boxes = T.detector_boxes(detector, detector_state, pre, device, args.threshold, args.tta)
        if head is not None and len(boxes):
            registered = T.align(pre, post, "keypoint")[0]
            features, roi = encode_boxes(encoder, pre, registered, boxes, device, args.encode_batch)
            keep = []
            with torch.no_grad():
                for start in range(0, len(features), 128):
                    tokens = torch.from_numpy(features[start:start + 128]).to(device)
                    weights = torch.from_numpy(roi[start:start + 128]).to(device).float()
                    valid = head(tokens, weights).float().sigmoid()[:, 0]
                    keep.append(valid.cpu().numpy())
            valid = np.concatenate(keep)
            boxes = boxes[valid >= args.validity_threshold]
        return boxes

    rows = json.loads(T.MANIFEST.read_bytes())[args.split]
    args.result.mkdir(parents=True, exist_ok=True)
    protocol = {"split": args.split, "scenes": len(rows), "detector": str(args.detector.resolve()),
                "classifier": str(args.classifier.resolve()), "tta": args.tta,
                "threshold": args.threshold,
                "validity_head": str(args.validity_head.resolve()) if args.validity_head else None,
                "validity_threshold": args.validity_threshold}
    (args.result / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluate(rows, grader, "hybrid_tta", None, args.result, known_boxes=False, pixel_audit=False,
             boxes_from=proposals, registration_method="keypoint")


if __name__ == "__main__":
    main()
