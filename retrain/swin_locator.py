"""Train a Swin-feature Faster R-CNN on pre-event building boxes."""

import argparse
from collections import OrderedDict
from functools import lru_cache
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import swin_t
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign, batched_nms

from detector import SOURCE_CACHE, building_boxes, match
from features import read_rgb
from inventory import ROOT, write_json

TILE = 256
WEIGHTS = ROOT / "weights/swin_t-704ceda3.pth"
OUT = ROOT / "run/swin_loc"


class SwinPyramid(nn.Module):
    out_channels = 128

    def __init__(self, pretrained):
        super().__init__()
        source = swin_t(weights=None)
        if pretrained:
            source.load_state_dict(torch.load(WEIGHTS, map_location="cpu"), strict=True)
        self.features = source.features
        self.features[0].requires_grad_(False)
        self.features[1].requires_grad_(False)
        self.features[2].requires_grad_(False)
        self.pyramid = FeaturePyramidNetwork([96, 192, 384, 768], self.out_channels)

    def forward(self, image):
        stages = OrderedDict()
        for i, layer in enumerate(self.features):
            image = layer(image)
            if i % 2:
                stages[str((i - 1) // 2)] = image.permute(0, 3, 1, 2).contiguous()
        return self.pyramid(stages)


def detector(pretrained, architecture="swin_t"):
    normalization = {}
    if architecture == "satlas_b":
        from satlas_features import SatlasPyramid

        backbone = SatlasPyramid(pretrained)
        anchors = AnchorGenerator(((32,), (64,), (128,), (256,)), ((.5, 1., 2.),) * 4)
        pool_size, representation, input_size = 7, 1024, 384
        normalization = {"image_mean": [0., 0., 0.], "image_std": [1., 1., 1.]}
    else:
        backbone = SwinPyramid(pretrained)
        anchors = AnchorGenerator(((12, 18), (24, 32), (45, 64), (90, 128)), ((.6, 1., 1.67),) * 4)
        pool_size, representation, input_size = 5, 384, 320
    roi_pool = MultiScaleRoIAlign(["0", "1", "2", "3"], output_size=pool_size, sampling_ratio=2)
    mlp = TwoMLPHead(backbone.out_channels * pool_size ** 2, representation)
    predictor = FastRCNNPredictor(representation, 2)
    model = FasterRCNN(backbone, num_classes=None, min_size=input_size, max_size=input_size,
                      rpn_anchor_generator=anchors, rpn_pre_nms_top_n_train=1200,
                      rpn_pre_nms_top_n_test=700, rpn_post_nms_top_n_train=540,
                      rpn_post_nms_top_n_test=430, box_roi_pool=roi_pool,
                      box_head=mlp, box_predictor=predictor, box_score_thresh=.08,
                      box_nms_thresh=.48, box_detections_per_img=130,
                      box_batch_size_per_image=192, **normalization)
    if architecture == "satlas_b" and pretrained:
        from satlas_features import initialize_building_head

        initialize_building_head(model)
    return model


@lru_cache(maxsize=16)
def scene(record_key, source):
    cache = SOURCE_CACHE / f"{record_key}.npy"
    if cache.is_file():
        image = np.load(cache, mmap_mode="r")[..., :3]
    else:
        image = read_rgb(Path(source) / "images" / f"{record_key}_pre_disaster.png")
    return image


def source_scene(record):
    return scene(record["chip"], record["source"])


def rgb_tensor(image, device):
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(image, -1, 0))).to(device).float().div_(255)


def crop_training(image, bounds, rng, tile=TILE):
    h, w = image.shape[:2]
    if min(h, w) < tile:
        image = np.pad(image, ((0, max(0, tile - h)), (0, max(0, tile - w)), (0, 0)), mode="edge")
        h, w = image.shape[:2]
    y0, x0 = int(rng.integers(h - tile + 1)), int(rng.integers(w - tile + 1))
    crop = image[y0:y0 + tile, x0:x0 + tile]
    translation = np.array([x0, y0, x0, y0], np.float32)
    boxes = bounds - translation
    clipped = boxes.clip(0, tile)
    area = np.maximum(1., (boxes[:, 2:] - boxes[:, :2]).prod(1))
    inside = clipped[:, 2:] - clipped[:, :2]
    keep = (inside.min(1) >= 3) & (inside.prod(1) >= area * .52)
    boxes = clipped[keep]
    if rng.random() < .5:
        crop = crop[:, ::-1]
        boxes[:, [0, 2]] = tile - boxes[:, [2, 0]]
    if rng.random() < .48:
        crop = crop[::-1]
        boxes[:, [1, 3]] = tile - boxes[:, [3, 1]]
    return crop, boxes


def auxiliary_crop(tiles, limit, rng, tile=TILE):
    image = tiles[rng.integers(limit)]
    _, _, stats, _ = cv2.connectedComponentsWithStats(image[..., 3].copy())
    boxes = []
    for x, y, width, height, area in stats[1:]:
        if min(width, height) >= 4 and max(width, height) <= 160 and area >= 18:
            boxes.append([x, y, x + width, y + height])
    return crop_training(image[..., :3], np.asarray(boxes, dtype=np.float32).reshape(-1, 4), rng, tile)


@torch.no_grad()
def swin_boxes(model, image, device, threshold=.16, batch=6, nms=.45, tile=TILE):
    height, width = image.shape[:2]
    if min(height, width) < tile:
        image = np.pad(image, ((0, max(0, tile - height)), (0, max(0, tile - width)), (0, 0)), mode="edge")
    h, w = image.shape[:2]
    stride = round(tile * 176 / TILE)
    edge = round(tile * 19 / TILE)
    rows = sorted({*range(0, h - tile + 1, stride), h - tile})
    cols = sorted({*range(0, w - tile + 1, stride), w - tile})
    windows = [(y, x) for y in rows for x in cols]
    predicted, confidences = [], []
    model.eval()
    for start in range(0, len(windows), batch):
        group = windows[start:start + batch]
        crops = [rgb_tensor(image[y:y + tile, x:x + tile], device) for y, x in group]
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            values = model(crops)
        for (top, left), item in zip(group, values):
            boxes, scores = item["boxes"].float(), item["scores"].float()
            cx, cy = ((boxes[:, :2] + boxes[:, 2:]) * .5).unbind(1)
            keep = ((scores >= threshold) & (cx >= (edge if left else 0)) &
                    (cx < (tile - edge if left + tile < w else width - left)) &
                    (cy >= (edge if top else 0)) &
                    (cy < (tile - edge if top + tile < h else height - top)))
            offset = boxes.new_tensor([left, top, left, top])
            predicted.append((boxes[keep] + offset).minimum(boxes.new_tensor([width, height, width, height])))
            confidences.append(scores[keep])
    boxes, scores = torch.cat(predicted), torch.cat(confidences)
    kept = batched_nms(boxes, scores, torch.zeros_like(scores, dtype=torch.int64), nms)
    return boxes[kept].cpu().numpy(), scores[kept].cpu().numpy()


def evaluate(model, validation, device, image_loader=source_scene, tile=TILE):
    cutoffs = [.12, .24, .36, .48, .60, .72, .83]
    counts = np.zeros((len(cutoffs), 3), np.int64)
    for record, ground in validation:
        image = image_loader(record)
        boxes, scores = swin_boxes(model, image, device, cutoffs[0], tile=tile)
        for i, threshold in enumerate(cutoffs):
            filtered = boxes[scores >= threshold]
            matched, _ = match(filtered, ground)
            counts[i] += [len(matched), len(filtered), len(ground)]
    f1 = 2 * counts[:, 0] / np.maximum(1, counts[:, 1:].sum(1))
    index = int(f1.argmax())
    return {"threshold": cutoffs[index], "box_f1_iou_50": float(f1[index]),
            "counts": counts[index].tolist()}


def save(model, out, row):
    interim = out / "detector.tmp"
    torch.save({"model": model.state_dict(), **row}, interim)
    interim.replace(out / "detector.pt")


def load_swin(path, device):
    state = torch.load(path, map_location="cpu")
    model = detector(False, state.get("architecture", "swin_t")).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), state


def train(args, device, datasets=None, image_loader=source_scene):
    torch.manual_seed(5123)
    rng = np.random.default_rng(41027)
    args.out.mkdir(parents=True, exist_ok=True)
    tile = getattr(args, "tile_size", TILE)
    architecture = getattr(args, "architecture", "swin_t")
    if datasets is None:
        data = json.loads((ROOT / "data/manifest.json").read_bytes())
        datasets = [[(r, building_boxes(r)) for r in data[split]] for split in ("train", "val")]
    train_rows, validation = datasets
    if args.init:
        model, original = load_swin(args.init, device)
        architecture = original.get("architecture", "swin_t")
        same_scale = original.get("tile_size", TILE) == tile
        best = original.get("box_f1_iou_50", -1.) if same_scale and image_loader is source_scene else -1.
    else:
        model = detector(True, architecture).to(device)
        best = -1.
    active = [p for p in model.parameters() if p.requires_grad]
    parameters = active
    if architecture == "satlas_b":
        backbone = [p for p in model.backbone.parameters() if p.requires_grad]
        ids = {id(p) for p in backbone}
        parameters = [{"params": backbone, "lr": args.lr * .24},
                      {"params": [p for p in active if id(p) not in ids], "lr": args.lr}]
    optim = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.015)
    eta = .00001 if architecture == "satlas_b" else .00003
    adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=eta)
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    started = time.monotonic()
    history = []
    if getattr(args, "validate_initial", False):
        row = {"epoch": 0, "tile_size": tile, "architecture": architecture,
               **evaluate(model, validation, device, image_loader, tile)}
        print(json.dumps(row), flush=True)
        if row["box_f1_iou_50"] > best:
            best = row["box_f1_iou_50"]
            save(model, args.out, row)
        history.append(row)
    if args.aux_buildings:
        folder = ROOT.parent / "cache"
        aux = np.load(folder / "building_train.npy", mmap_mode="r")
        aux_count = len(json.loads((folder / "building.json").read_bytes())["train"]["names"])
    else:
        aux, aux_count = None, 0
    for epoch in range(1, args.epochs + 1):
        total = 0.
        steps = 0
        model.train()
        for _ in range(args.steps):
            images, targets = [], []
            for index in rng.integers(len(train_rows), size=args.batch):
                if aux is not None and rng.random() < args.aux_buildings:
                    image, b = auxiliary_crop(aux, aux_count, rng, tile)
                else:
                    record, boxes = train_rows[index]
                    image, b = crop_training(image_loader(record), boxes, rng, tile)
                x = rgb_tensor(image, device)
                x = (x * rng.uniform(.83, 1.17) + rng.uniform(-.05, .05)).clamp(0, 1)
                images.append(x)
                targets.append({"boxes": torch.from_numpy(np.ascontiguousarray(b)).to(device),
                                "labels": torch.ones(len(b), device=device, dtype=torch.int64)})
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {losses}")
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(active, 8.)
            scaler.step(optim)
            scaler.update()
            total += float(loss.detach())
            steps += 1
            if time.monotonic() - started >= args.seconds:
                break
        adjust.step()
        row = {"epoch": epoch, "loss": total / steps, "steps": steps,
               "elapsed": time.monotonic() - started, "tile_size": tile, "architecture": architecture}
        stop = time.monotonic() - started >= args.seconds
        if epoch == 1 or epoch % 3 == 0 or epoch == args.epochs or stop:
            row.update(evaluate(model, validation, device, image_loader, tile))
            if row["box_f1_iou_50"] > best:
                best = row["box_f1_iou_50"]
                save(model, args.out, row)
        history.append(row)
        write_json(args.out / "history.json", history)
        print(json.dumps(row), flush=True)
        if stop:
            break


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--epochs", type=int, default=15)
    cli.add_argument("--batch", type=int, default=4)
    cli.add_argument("--steps", type=int, default=110)
    cli.add_argument("--lr", type=float, default=.00021)
    cli.add_argument("--seconds", type=int, default=2050)
    cli.add_argument("--out", type=Path, default=OUT)
    cli.add_argument("--init", type=Path)
    cli.add_argument("--aux-buildings", type=float, default=0.)
    cli.add_argument("--tile-size", type=int, choices=(160, 192, 224, 256, 320), default=TILE)
    cli.add_argument("--architecture", choices=("swin_t", "satlas_b"), default="swin_t")
    cli.add_argument("--validate-initial", action="store_true")
    args = cli.parse_args()
    if min(args.epochs, args.batch, args.steps, args.seconds, args.lr) <= 0:
        cli.error("training parameters must be positive")
    if not 0 <= args.aux_buildings < 1:
        cli.error("auxiliary sample fraction must be between 0 and 1")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
