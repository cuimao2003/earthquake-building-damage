"""A paired roof Transformer with proposal validity and ordinal damage outputs."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dino import PATCH
from features import patch_pair


PATCH_SIZE = 168
GRID = 6


@torch.no_grad()
def encode_roofs(encoder, rgb, masks, device):
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 1, 4, 2, 3))).to(device)
    mask = torch.from_numpy(np.ascontiguousarray(masks)).to(device)
    pixels = tensor.flatten(0, 1).float() / 255
    mean = pixels.new_tensor([.485, .456, .406])[None, :, None, None]
    scale = pixels.new_tensor([.229, .224, .225])[None, :, None, None]
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        features = encoder((pixels - mean) / scale)
    height, width = pixels.shape[-2] // PATCH, pixels.shape[-1] // PATCH
    spatial = features[:, 1:].float().transpose(1, 2).reshape(len(pixels), encoder.width, height, width)
    spatial = F.adaptive_avg_pool2d(spatial, GRID).flatten(2).transpose(1, 2)
    tokens = torch.cat((features[:, :1].float(), spatial), dim=1).reshape(len(tensor), 2, -1, encoder.width)
    weights = F.adaptive_avg_pool2d(mask[:, None].float(), GRID).flatten(1)
    return tokens.half().cpu().numpy(), weights.half().cpu().numpy()


def encode_boxes(encoder, before, after, boxes, device, batch):
    visual, roi = [], []
    for start in range(0, len(boxes), batch):
        views, masks = zip(*(patch_pair(before, after, box, size=PATCH_SIZE) for box in boxes[start:start + batch]))
        features, weights = encode_roofs(encoder, np.stack(views), np.stack(masks), device)
        visual.append(features)
        roi.append(weights)
    if not visual:
        return (np.empty((0, 2, 1 + GRID * GRID, encoder.width), np.float16),
                np.empty((0, GRID * GRID), np.float16))
    return np.concatenate(visual), np.concatenate(roi)


def roi_mean(tokens, weight):
    expanded = weight.unsqueeze(-1)
    return (tokens * expanded).sum(1) / expanded.sum(1).clamp_min(.001)


class RoofTransformer(nn.Module):
    def __init__(self, width, variant="attention", hidden=96):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU())
        self.temporal = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.position = nn.Parameter(torch.zeros(1, GRID * GRID, hidden))
        nn.init.normal_(self.position, std=.02)
        self.variant = variant
        if variant == "attention":
            self.align = nn.MultiheadAttention(hidden, 4, dropout=.10, batch_first=True)
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=.15,
                                               activation="gelu", batch_first=True)
            self.context = nn.TransformerEncoder(layer, 2)
            slots = torch.arange(GRID * GRID)
            coords = torch.stack((torch.div(slots, GRID, rounding_mode="floor"), slots % GRID), dim=1)
            distant = (coords[:, None] - coords[None]).abs().amax(-1) > 1
            self.register_buffer("distant", distant, persistent=False)
        self.summary = nn.Sequential(nn.Linear(6 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(.30))
        self.outputs = nn.Linear(hidden, 4)

    def forward(self, tokens, weight):
        before, after = self.projection(tokens.float()).unbind(1)
        a, b = before[:, 1:], after[:, 1:]
        if self.variant == "attention":
            a = a + self.align(b + self.position, a + self.position, a,
                               attn_mask=self.distant, need_weights=False)[0]
        diff = b - a
        spatial = self.temporal(torch.cat((a, b, diff, diff.abs()), dim=-1)) + self.position
        if self.variant == "attention":
            spatial = self.context(spatial)
        cls_a, cls_b = before[:, 0], after[:, 0]
        global_change = cls_b - cls_a
        summary = torch.cat((roi_mean(spatial, weight), roi_mean(spatial, 1. - weight),
                             cls_a, cls_b, global_change, global_change.abs()), dim=1)
        return self.outputs(self.summary(summary))


def probabilities(logits):
    valid, damage, severe, _ = logits.float().sigmoid().unbind(1)
    grades = torch.stack((1 - damage, damage * (1 - severe), damage * severe), dim=1)
    return valid, grades


def training_loss(logits, target, has_background):
    eligible = target >= 0
    damaged = target > 0
    def binary(channel, keep, labels):
        return F.binary_cross_entropy_with_logits(logits[keep, channel], labels[keep].float(), reduction="sum") / keep.sum().clamp_min(1)
    damage = binary(1, eligible, damaged)
    severe = binary(2, damaged, target == 2)
    if not has_background:
        return damage + severe
    weight = torch.where(eligible, 1., .45)
    validity = F.binary_cross_entropy_with_logits(logits[:, 0], eligible.float(), weight=weight)
    return damage + severe + .60 * validity
