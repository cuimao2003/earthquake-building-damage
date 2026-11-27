"""Independent KATE-CD (Turkey 2023) check at pixel level.

KATE-CD ships 512x512 pre/post pairs with a binary damage mask (no building boxes),
so we rasterize the model's predicted damaged building boxes and compare the binary
damage mask with the label mask. This is a cross-event sanity check, not a replacement
for building-level xBD metrics.

Usage:
  python evaluate_kate.py --mode proposal --out run/mix_palu_pooled --manifest data/manifest_plus_palu.json \
      --result run/kate_mix_palu_pooled
  python evaluate_kate.py --mode old --result run/kate_old
"""

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch

import train_proposal_transformer as T
from infer_retrain import SelectedGrader, load_localizers
from inventory import ROOT, write_json

KATE = Path(r"E:/datasets/kate-cd")


def decode(row, key):
    image = cv2.imdecode(np.frombuffer(row[key]["bytes"], np.uint8), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(image[:, :, ::-1])


def mask_of(row):
    mask = np.asarray(Image.open(io.BytesIO(row["label"]["bytes"])).convert("L"))
    return mask > 127


def rasterize(shape, boxes, grades):
    canvas = np.zeros(shape, np.uint8)
    for box, grade in zip(boxes, grades):
        if grade <= 0:
            continue
        left, top, right, bottom = box
        cv2.rectangle(canvas, (int(left), int(top)), (int(right), int(bottom)), 1, thickness=-1)
    return canvas.astype(bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--mode", choices=("old", "proposal"), default="proposal")
    parser.add_argument("--out", type=Path, help="proposal mode: trained head run directory")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/dinov2_vitb14_pretrain.pth")
    parser.add_argument("--detector", type=Path, default=ROOT / "run/swin_loc_mix/detector.pt")
    parser.add_argument("--classification", type=Path, default=ROOT / "run/mexico_large_selection")
    parser.add_argument("--reference", type=Path, default=ROOT / "run/mexico_large_selection")
    parser.add_argument("--encode-batch", type=int, default=24)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = pq.read_table(KATE / f"{args.split}-00000-of-00001.parquet").to_pylist()
    args.result.mkdir(parents=True, exist_ok=True)

    if args.mode == "old":
        grader = SelectedGrader(args.classification, device)
        mode = grader.calibrated_localizer
        localizers = load_localizers(device)
    else:
        checkpoint = torch.load(args.out / "head.pt", map_location="cpu")
        if checkpoint["cache"]["manifest_sha256"] != T.digest(args.manifest):
            raise ValueError("manifest does not match the trained head")
        wrapper = SimpleNamespace(out=args.out, fusion=None, weights=args.weights,
                                  detector=args.detector, encode_batch=args.encode_batch)
        grader = T.ProposalGrader(wrapper, device)

    tp = fp = fn = 0
    chips = []
    for index, row in enumerate(rows, 1):
        pre, post = decode(row, "pre_image"), decode(row, "post_image")
        label = mask_of(row)
        if args.mode == "old":
            from infer_retrain import localize
            boxes = localize(mode, localizers, pre, post, grader.device)
            estimates, _ = grader.classify(pre, post, boxes)
        else:
            boxes = grader.propose(pre, post)
            estimates, _ = grader.classify(pre, post, boxes)
        prediction = rasterize(label.shape, boxes, estimates)
        tp += int((prediction & label).sum())
        fp += int((prediction & ~label).sum())
        fn += int((~prediction & label).sum())
        chips.append({"chip": index, "boxes": int(len(boxes)), "damaged_boxes": int((estimates > 0).sum()),
                      "label_pixels": int(label.sum()), "predicted_pixels": int(prediction.sum())})
        if index % 10 == 0 or index == len(rows):
            print(f"KATE {args.split}: {index}/{len(rows)}", flush=True)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    result = {"split": args.split, "mode": args.mode, "chips": len(rows),
              "pixel_damage_precision": precision, "pixel_damage_recall": recall,
              "pixel_damage_f1": f1, "iou": tp / max(1, tp + fp + fn),
              "true_positive_pixels": tp, "false_positive_pixels": fp, "false_negative_pixels": fn,
              "per_chip": chips}
    write_json(args.result / "kate_eval.json", result)
    print(json.dumps({k: result[k] for k in ("split", "mode", "chips", "pixel_damage_precision",
                                             "pixel_damage_recall", "pixel_damage_f1", "iou")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
