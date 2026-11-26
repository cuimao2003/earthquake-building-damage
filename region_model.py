"""Building-centered damage scoring on footprints proposed by the dense model."""

from functools import lru_cache

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import conv


CHIP_SIZE = 96


class RegionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pyramid = nn.ModuleList([conv(3, 16, 2), conv(16, 24, 2), conv(24, 40, 2)])
        self.fuse = nn.Conv2d(120, 48, 1)
        self.position = nn.Conv2d(48, 48, 3, padding=1, groups=48)
        self.region_position = nn.Conv2d(1, 48, 1, bias=False)
        self.attention = nn.ModuleList(
            nn.TransformerEncoderLayer(48, 4, dim_feedforward=96, dropout=0.14,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(2)
        )
        self.head = nn.Sequential(nn.LayerNorm(96), nn.Dropout(0.18), nn.Linear(96, 48),
                                  nn.GELU(), nn.Linear(48, 3))

    def forward(self, before, after, region):
        x = torch.cat((before, after))
        for layer in self.pyramid:
            x = layer(x)
        pre, post = x.chunk(2)
        x = self.fuse(torch.cat((pre, post, post - pre), dim=1))
        core = F.adaptive_avg_pool2d(region, x.shape[-2:])
        x = x + self.position(x) + self.region_position(core)
        tokens = x.flatten(2).transpose(1, 2)
        for layer in self.attention:
            tokens = layer(tokens)
        weights = core.flatten(2).transpose(1, 2).float()
        local = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.)
        return self.head(torch.cat((local, tokens.mean(dim=1)), dim=1))


def building_crop(before, after, bounds, rng=None):
    x0, y0, x1, y1 = bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0)
    extent = float(np.clip(side * 2.05, 44, 174))
    if rng is not None:
        extent *= rng.uniform(0.80, 1.24)
        cx += rng.uniform(-0.13, 0.13) * extent
        cy += rng.uniform(-0.13, 0.13) * extent
    left, top = cx - extent / 2, cy - extent / 2
    scale = CHIP_SIZE / extent
    height, width = before.shape[:2]
    x0_src, x1_src = crop_limits(left, extent, width)
    y0_src, y1_src = crop_limits(top, extent, height)
    mapping = np.float32([[scale, 0, (x0_src - left) * scale],
                          [0, scale, (y0_src - top) * scale]])
    views = [cv2.warpAffine(np.ascontiguousarray(im[y0_src:y1_src, x0_src:x1_src]),
                            mapping, (CHIP_SIZE, CHIP_SIZE),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
             for im in (before, after)]
    region = np.zeros((CHIP_SIZE, CHIP_SIZE), np.uint8)
    start = (round((x0 - left) * scale), round((y0 - top) * scale))
    stop = (round((x1 - left) * scale), round((y1 - top) * scale))
    cv2.rectangle(region, start, stop, 1, thickness=-1)
    return np.dstack([*views, region])


def crop_limits(origin, extent, length):
    start = int(np.floor(origin)) - 2
    stop = int(np.ceil(origin + extent)) + 3
    # Include the source pixels needed to reflect past an image edge.
    return max(0, min(start, 2 * length - stop - 2)), min(length, max(stop, -start + 2))


@lru_cache(maxsize=8)
def read_pair(path):
    chip = np.load(path, mmap_mode="r")
    return chip[..., :3], chip[..., 3:6]


def tensor_inputs(array, device):
    data = torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2))).to(device)
    rgb = data[:, :6].float().div_(127.5).sub_(1.)
    return rgb[:, :3], rgb[:, 3:6], data[:, 6:].float()


def extract_proposals(footprint):
    foreground = (footprint >= 0.5).astype(np.uint8)
    if not foreground.any():
        return np.zeros(footprint.shape, np.int32), np.zeros((0, 4), np.float32)
    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    seeds = (distance >= cv2.dilate(distance, kernel) - 1e-4) & (distance >= 2.)
    count, components, stats, _ = cv2.connectedComponentsWithStats(foreground)
    represented = set(np.unique(components[seeds]).tolist())
    for i in range(1, count):
        if i in represented:
            continue
        x, y, width, height, _ = stats[i]
        area = np.s_[y:y + height, x:x + width]
        _, _, _, (px, py) = cv2.minMaxLoc(footprint[area], mask=(components[area] == i).astype(np.uint8))
        seeds[y + py, x + px] = True
    _, nearest = cv2.distanceTransformWithLabels((~seeds).astype(np.uint8), cv2.DIST_L2, 5,
                                                  labelType=cv2.DIST_LABEL_CCOMP)
    instance = nearest * foreground
    y, x = np.nonzero(instance)
    index = instance[y, x]
    n = int(index.max()) + 1
    left = np.full(n, footprint.shape[1], np.int32)
    top = np.full(n, footprint.shape[0], np.int32)
    right = np.zeros(n, np.int32)
    bottom = np.zeros(n, np.int32)
    np.minimum.at(left, index, x)
    np.minimum.at(top, index, y)
    np.maximum.at(right, index, x)
    np.maximum.at(bottom, index, y)
    return instance, np.stack((left[1:], top[1:], right[1:] + 1, bottom[1:] + 1), axis=1)


def dense_region_scores(probs, instances, count):
    flat = instances.ravel()
    sizes = np.bincount(flat, minlength=count + 1).clip(1)
    scoring = np.stack([np.bincount(flat, weights=p.ravel(), minlength=count + 1) / sizes
                        for p in probs[1:]], axis=1)
    return scoring.astype(np.float32)


def proposal_crops(before, after, boxes):
    chips = np.empty((len(boxes), CHIP_SIZE, CHIP_SIZE, 7), np.uint8)
    for i, box in enumerate(boxes):
        chips[i] = building_crop(before, after, box)
    return chips


def harmonize_after(before, after, region):
    # Use context outside the proposed roof to estimate illumination changes.
    context = 1 - region
    size = context.sum(dim=(2, 3), keepdim=True).clamp_min(1.)
    mean_before = (before * context).sum(dim=(2, 3), keepdim=True) / size
    mean_after = (after * context).sum(dim=(2, 3), keepdim=True) / size
    var_before = ((before - mean_before).square() * context).sum(dim=(2, 3), keepdim=True) / size
    var_after = ((after - mean_after).square() * context).sum(dim=(2, 3), keepdim=True) / size
    gain = ((var_before + .005) / (var_after + .005)).sqrt().clamp_(.65, 1.55)
    corrected = (after - mean_after) * gain + mean_before
    return torch.where(size > 32, (.25 * after + .75 * corrected).clamp_(-1, 1), after)


@torch.no_grad()
def score_crops(model, crops, device, batch=64, harmonize=False):
    model.eval()
    scores = np.zeros((len(crops) + 1, 3), np.float32)
    for start in range(0, len(crops), batch):
        pre, post, roi = tensor_inputs(crops[start:start + batch], device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            probs = model(pre, post, roi).float().softmax(dim=1)
            if harmonize:
                compensated = model(pre, harmonize_after(pre, post, roi), roi).float().softmax(dim=1)
                probs = .5 * (probs + compensated)
        scores[start + 1:start + 1 + len(pre)] = probs.cpu().numpy()
    return scores


def region_scores(scores, dense_scores, blend):
    return (1 - blend) * scores + blend * dense_scores


def instance_mask(instances, scores, thresholds):
    proposals = np.ones(len(scores), np.uint8)
    proposals[0] = 0
    minor = (scores[1:, 1] >= scores[1:, 2]) & (scores[1:, 1] >= thresholds["minor"])
    severe = (scores[1:, 2] > scores[1:, 1]) & (scores[1:, 2] >= thresholds["severe"])
    proposals[1:][minor] = 2
    proposals[1:][severe] = 3
    return proposals[instances]


def load_region_model(path, device):
    state = torch.load(path, map_location=device)
    if state.get("stage") != "building-regions":
        raise ValueError("Expected a building-region checkpoint")
    model = RegionTransformer().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def predict_regions(model, state, before, after, dense, device):
    instances, boxes = extract_proposals(dense[0])
    crops = proposal_crops(before, after, boxes)
    scores = score_crops(model, crops, device, harmonize=state.get("harmonize", False))
    dense_scores = dense_region_scores(dense, instances, len(boxes))
    blended = region_scores(scores, dense_scores, state["blend"])
    return instance_mask(instances, blended, state["thresholds"])
