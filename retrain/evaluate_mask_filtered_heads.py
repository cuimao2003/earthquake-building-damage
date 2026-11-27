"""Mask-filtered TTA proposals graded by the proposal-Transformer head (end-to-end audit).

For each validation scene:
1. building mask from the fine-tuned UNet (xBD domain);
2. flip-TTA + WBF proposals from the Swin detector;
3. filter proposals by mean building-mask probability inside the box (and optional validity);
4. grade with the p1_tta_pooled RoI head (thresholds from its validation calibration);
5. report localization and 3-class macro F1 for a grid of mask thresholds.

Usage:
  python evaluate_mask_filtered_heads.py --split val --head run/p1_tta_pooled \
      --result run/mask_filtered_val.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_proposal_transformer as T
from classifier import predict_grade
from eval_building_unet import infer_mask
from infer_retrain import grade_metrics, ground_truth, paired_images
from inventory import ROOT, write_json
from roi_transformer import RoofTransformer, encode_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--head", type=Path, default=ROOT / "run/p1_tta_pooled")
    parser.add_argument("--mask-checkpoint", type=Path,
                        default=ROOT / "run/building_unet_xbd/building_unet_xbd.pt")
    parser.add_argument("--detector", type=Path, default=ROOT / "run/swin_loc_mix/detector.pt")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/dinov2_vitb14_pretrain.pth")
    parser.add_argument("--threshold", type=float, default=0.36)
    parser.add_argument("--tta", choices=("none", "h"), default="h")
    parser.add_argument("--encode-batch", type=int, default=24)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from train_building_unet import UNet
    mask_state = torch.load(args.mask_checkpoint, map_location="cpu")
    mask_model = UNet().to(device)
    mask_model.load_state_dict(mask_state["model"])
    mask_model.eval()
    detector, detector_state = T.load_swin(args.detector, device)
    checkpoint = torch.load(args.head / "head.pt", map_location="cpu")
    head = RoofTransformer(checkpoint["width"], checkpoint["variant"]).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    calibration = json.loads((args.head / "validation.json").read_bytes())
    thresholds = np.asarray(calibration["thresholds"], np.float32)
    validity_threshold = float(calibration.get("validity_threshold", 0.0))
    encoder = T.load_encoder(args.weights, device)
    records = json.loads(T.MANIFEST.read_bytes())[args.split]
    support = np.zeros(3, np.int64)
    variants = {}
    grid = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
    for variant in ("mask", "mask_validity"):
        for value in grid:
            variants[f"{variant}_{value}"] = {"table": np.zeros((3, 3), np.int64),
                                              "missed": np.zeros(3, np.int64),
                                              "extra": np.zeros(3, np.int64),
                                              "boxes": 0, "matched": 0, "truth": 0}
    for index, row in enumerate(records, 1):
        pre, post = paired_images(row)
        truth, grades = ground_truth(row)
        support += np.bincount(grades[grades >= 0], minlength=3)
        mask = infer_mask(mask_model, pre, device)
        proposals = T.detector_boxes(detector, detector_state, pre, device, args.threshold, args.tta)
        if len(proposals) == 0:
            continue
        coverage = np.asarray([mask[max(0, int(b[1])):int(b[3]), max(0, int(b[0])):int(b[2])].mean()
                               if int(b[2]) > int(b[0]) and int(b[3]) > int(b[1]) else 0.
                               for b in proposals], np.float32)
        registered = T.align(pre, post, "keypoint")[0]
        features, roi = encode_boxes(encoder, pre, registered, proposals, device, args.encode_batch)
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(features), 128):
                tokens = torch.from_numpy(features[start:start + 128]).to(device)
                weights = torch.from_numpy(roi[start:start + 128]).to(device).float()
                probabilities.append(head(tokens, weights).float().sigmoid().cpu().numpy())
        probability = np.concatenate(probabilities)
        damage, severe = probability[:, 1], probability[:, 2]
        scores = np.stack((1 - damage, damage * (1 - severe), damage * severe), axis=1)
        validity = probability[:, 0]
        matched, assigned = T.match(proposals, truth) if hasattr(T, "match") else (None, None)
        from detector import match
        matched, assigned = match(proposals, truth)
        assigned_mask = np.ones(len(truth), bool)
        assigned_mask[assigned] = False
        keep_grade = grades[assigned] >= 0
        matched_mask = np.zeros(len(proposals), bool)
        matched_mask[matched] = True
        for variant_name, threshold in (("mask", grid), ("mask_validity", grid)):
            for value in threshold:
                selected = coverage >= value
                graded = predict_grade(scores, thresholds)
                if variant_name == "mask_validity":
                    selected = selected & (validity >= validity_threshold)
                entry = variants[f"{variant_name}_{value}"]
                entry["boxes"] += int(selected.sum())
                entry["truth"] += int(len(truth))
                np.add.at(entry["table"], (grades[assigned[keep_grade]], graded[matched[keep_grade]]), 1)
                entry["missed"] += np.bincount(grades[assigned_mask & (grades >= 0)], minlength=3)
                unmatched = (~matched_mask) & selected
                entry["extra"] += np.bincount(graded[unmatched], minlength=3)
                entry["matched"] += int((matched_mask & selected).sum())
        if index % 5 == 0 or index == len(records):
            print(f"mask-filtered audit {index}/{len(records)}", flush=True)
    output = {"split": args.split, "threshold": args.threshold, "tta": args.tta,
              "head": str(args.head.resolve()), "head_thresholds": thresholds.tolist(),
              "head_validity_threshold": validity_threshold, "support": support.tolist(), "variants": {}}
    for name, entry in variants.items():
        result = grade_metrics(entry["table"], entry["missed"], entry["extra"])
        precision = entry["matched"] / max(1, entry["boxes"])
        recall = entry["matched"] / max(1, entry["truth"])
        output["variants"][name] = {
            "mask_threshold": float(name.rsplit("_", 1)[1]), "boxes": entry["boxes"],
            "matched": entry["matched"],
            "localization_precision": round(precision, 4), "localization_recall": round(recall, 4),
            "localization_f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
            "macro_f1": round(result["macro_f1"], 4),
            "damaged_macro_f1": round(result["damaged_macro_f1"], 4),
            "accuracy": round(result["accuracy"], 4),
            "classes": {k: {"f1": round(v["f1"], 4), "predicted": v["predicted"]}
                        for k, v in result["classes"].items()},
        }
        print(name, json.dumps(output["variants"][name], ensure_ascii=False), flush=True)
    write_json(args.result, output)


if __name__ == "__main__":
    main()
