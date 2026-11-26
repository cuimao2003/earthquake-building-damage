"""Predict a footprint and a damage map from two registered RGB images."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from evaluation import classify, predict_map
from model import DamageTransformer
from prepare import CLASSES, decode, write_json
from region_model import load_region_model, predict_regions


RUN = Path(__file__).resolve().parent / "run"
COLORS = np.array([(25, 30, 39), (82, 166, 236), (255, 200, 59), (240, 69, 84)], np.uint8)
OPACITY = np.array([0, 0.26, 0.72, 0.78], np.float32)


def load_model(path, device):
    model = DamageTransformer().to(device)
    state = torch.load(path, map_location=device)
    if state.get("stage") != "earthquake" or tuple(state.get("classes", ())) != CLASSES:
        raise ValueError("This command requires a paired-image damage checkpoint")
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def overlay(image, mask):
    opacity = OPACITY[mask, None]
    return np.round(image * (1 - opacity) + COLORS[mask] * opacity).astype(np.uint8)


def save_prediction(out, image, mask, checkpoint, region_checkpoint=None):
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out / "damage_mask.png")
    Image.fromarray((mask != 0).astype(np.uint8) * 255).save(out / "footprint.png")
    Image.fromarray(COLORS[mask]).save(out / "damage_colors.png")
    Image.fromarray(overlay(image, mask)).save(out / "damage_overlay.png")
    count = np.bincount(mask.ravel(), minlength=4)
    info = {"shape": list(mask.shape), "checkpoint": str(checkpoint),
            "classes": CLASSES, "pixels": count.tolist()}
    if region_checkpoint is not None:
        info["region_checkpoint"] = str(region_checkpoint)
    write_json(out / "prediction.json", info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=RUN / "damage_model.pt")
    parser.add_argument("--region-checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=RUN / "prediction")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    pre, _ = decode(args.pre, cv2.IMREAD_COLOR)
    post, _ = decode(args.post, cv2.IMREAD_COLOR)
    pre, post = (np.ascontiguousarray(im[:, :, ::-1]) for im in (pre, post))
    probs = predict_map(model, pre, post, device, args.batch, amp=device.type == "cuda",
                        **checkpoint.get("inference", {}))
    if args.region_checkpoint is not None:
        region, settings = load_region_model(args.region_checkpoint, device)
        mask = predict_regions(region, settings, pre, post, probs, device)
    else:
        mask = classify(probs, checkpoint["thresholds"])
    save_prediction(args.out, post, mask, args.checkpoint.resolve(), args.region_checkpoint)
    print(args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
