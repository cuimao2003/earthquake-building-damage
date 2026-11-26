"""Extract Mexico building descriptors with the aerial SatlasPretrain Swin V2.

The window layout and V2 attention follow torchvision's BSD-licensed Swin
implementation. The local implementation bridges torchvision 0.13 to the
published checkpoint's cosine attention and post-normalization.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.swin_transformer import (
    ShiftedWindowAttention, SwinTransformer, SwinTransformerBlock, _patch_merging_pad,
)

from classifier import RUN, load_data, train_head, train_texture, choose
from finetune_mexico import STORE
from finetune_turkey import Patches
from inventory import ROOT


WEIGHTS = ROOT / "weights/satlas_aerial_swinb.pth"
OUT = RUN / "mexico_satlas"
DATA = ROOT / "data/features"


def window_partition(x, height, width, wh, ww):
    return x.reshape(-1, height // wh, wh, width // ww, ww, x.shape[-1]).permute(0, 1, 3, 2, 4, 5)


class AerialAttention(ShiftedWindowAttention):
    def __init__(self, dim, window_size, shift_size, num_heads, **options):
        super().__init__(dim, window_size, shift_size, num_heads, **options)
        del self.relative_position_bias_table
        self.logit_scale = nn.Parameter(torch.full((num_heads, 1, 1), math.log(10.)))
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512), nn.ReLU(), nn.Linear(512, num_heads, bias=False))
        wh, ww = window_size
        h, w = torch.meshgrid(torch.arange(1 - wh, wh), torch.arange(1 - ww, ww), indexing="ij")
        grid = torch.stack((h / (wh - 1), w / (ww - 1)), dim=-1).unsqueeze(0) * 8
        self.register_buffer("relative_coords_table", grid.sign() * (grid.abs() + 1).log2() / 3)

    def forward(self, value):
        batch, height, width, channels = value.shape
        wh, ww = self.window_size
        value = F.pad(value, (0, 0, 0, (-width) % ww, 0, (-height) % wh))
        _, ph, pw, _ = value.shape
        sh = self.shift_size[0] if ph > wh else 0
        sw = self.shift_size[1] if pw > ww else 0
        shifted = bool(sh or sw)
        if shifted:
            value = torch.roll(value, (-sh, -sw), (1, 2))
        cells = wh * ww
        windows = window_partition(value, ph, pw, wh, ww).reshape(-1, cells, channels)
        bias = self.qkv.bias
        if bias is not None:
            bias = torch.cat((bias[:channels], torch.zeros_like(bias[:channels]), bias[2 * channels:]))
        qkv = F.linear(windows, self.qkv.weight, bias)
        q, k, v = qkv.reshape(-1, cells, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2)
        table = self.cpb_mlp(self.relative_coords_table).reshape(-1, self.num_heads)
        relative = table[self.relative_position_index].reshape(cells, cells, self.num_heads).permute(2, 0, 1)
        attn = attn * self.logit_scale.clamp(max=math.log(100)).exp() + 16 * relative.sigmoid()
        if shifted:
            # Distinguish strips at a cyclic image edge, while keeping interior windows unchanged.
            py, px = torch.arange(ph, device=value.device), torch.arange(pw, device=value.device)
            yy = (py >= ph - wh).to(torch.int64) + (py >= ph - sh).to(torch.int64) if sh else torch.zeros_like(py)
            xx = (px >= pw - ww).to(torch.int64) + (px >= pw - sw).to(torch.int64) if sw else torch.zeros_like(px)
            strips = 3 * yy[:, None] + xx[None]
            masks = window_partition(strips[None, ..., None], ph, pw, wh, ww).reshape(-1, cells)
            separate = masks[:, :, None] != masks[:, None, :]
            attn = attn.reshape(batch, -1, self.num_heads, cells, cells)
            attn = (attn - 100. * separate[None, :, None]).reshape(-1, self.num_heads, cells, cells)
        attn = F.dropout(attn.softmax(-1), self.attention_dropout, self.training)
        value = self.proj((attn @ v).transpose(1, 2).reshape(-1, cells, channels))
        value = F.dropout(value, self.dropout, self.training)
        value = value.reshape(batch, ph // wh, pw // ww, wh, ww, channels)
        value = value.permute(0, 1, 3, 2, 4, 5).reshape(batch, ph, pw, channels)
        if shifted:
            value = torch.roll(value, (sh, sw), (1, 2))
        return value[:, :height, :width].contiguous()


class AerialBlock(SwinTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, attn_layer=AerialAttention)

    def forward(self, value):
        value = value + self.stochastic_depth(self.norm1(self.attn(value)))
        return value + self.stochastic_depth(self.norm2(self.mlp(value)))


class AerialMerge(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.reduction = nn.Linear(4 * width, 2 * width, bias=False)
        self.norm = nn.LayerNorm(2 * width)

    def forward(self, value):
        value = _patch_merging_pad(value)
        merged = torch.cat((value[..., 0::2, 0::2, :], value[..., 1::2, 0::2, :],
                            value[..., 0::2, 1::2, :], value[..., 1::2, 1::2, :]), dim=-1)
        return self.norm(self.reduction(merged))


def backbone(device):
    instance = SwinTransformer(patch_size=[4, 4], embed_dim=128, depths=[2, 2, 18, 2],
                               num_heads=[4, 8, 16, 32], window_size=[8, 8], block=AerialBlock)
    for index, width in ((2, 128), (4, 256), (6, 512)):
        instance.features[index] = AerialMerge(width)
    state = torch.load(WEIGHTS, map_location="cpu")
    prefix = "backbone.backbone.features."
    features = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    instance.features.load_state_dict(features, strict=True)
    return instance.features[:4].eval().requires_grad_(False).to(device)


@torch.no_grad()
def embed(model, rgb, roi):
    batch = rgb.shape[0]
    current = F.interpolate(rgb.flatten(0, 1).float() / 255., (128, 128), mode="bilinear", align_corners=False)
    scenes, roofs, similarities = [], [], []
    for index, layer in enumerate(model):
        current = layer(current)
        if index not in (1, 3):
            continue
        _, h, w, channels = current.shape
        features = F.normalize(current.float(), dim=-1).reshape(batch, 2, h, w, channels)
        keep = F.interpolate(roi[:, None].float(), (h, w), mode="area")[:, 0]
        weights = keep[:, None, :, :, None]
        scene = F.normalize(features.mean((2, 3)), dim=-1)
        roof = F.normalize((features * weights).sum((2, 3)), dim=-1)
        scenes.append(scene)
        roofs.append(roof)
        same = (features[:, 0] * features[:, 1]).sum(-1)
        outside = 1 - keep
        similarities.extend(((same * keep).sum((1, 2)) / keep.sum((1, 2)).clamp_min(.001),
                             (same * outside).sum((1, 2)) / outside.sum((1, 2)).clamp_min(.001),
                             (scene[:, 0] * scene[:, 1]).sum(-1)))
    appearance = torch.cat((torch.cat(scenes, dim=-1), torch.cat(roofs, dim=-1)), dim=1)
    change = torch.stack(similarities, dim=1)
    return appearance, change


def cache(split, model, device):
    output = DATA / f"{split}_satlas.npz"
    if output.is_file():
        with np.load(output, allow_pickle=False) as z:
            return {name: z[name] for name in z.files}
    examples = Patches(split, store=STORE)
    original = load_data(split)
    if not np.array_equal(examples.chip, original["chip"]) or not np.array_equal(examples.grade, original["grade"]):
        raise ValueError("Patch cache and embedding IDs differ")
    visual, numeric = [], []
    for offset in range(0, len(examples.grade), 64):
        ids = np.arange(offset, min(offset + 64, len(examples.grade)))
        rgb, roi, texture, _ = examples.batch(ids, device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            appearance, change = embed(model, rgb, roi)
        visual.append(appearance.cpu().numpy().astype(np.float16))
        numeric.append(torch.cat((texture, change), dim=1).cpu().numpy())
        if offset % 2048 == 0 or offset + 64 >= len(examples.grade):
            print(f"Satlas crops {split}: {offset + len(ids)}/{len(examples.grade)}", flush=True)
    data = {"appearance": np.concatenate(visual), "change": np.concatenate(numeric),
            "grade": original["grade"], "chip": original["chip"], "uid": original["uid"]}
    np.savez(output, **data)
    return data


if __name__ == "__main__":
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = backbone(device)
    train, val = [cache(split, model, device) for split in ("train", "val")]
    del model
    OUT.mkdir(exist_ok=True)
    head = train_head(train, val, device, OUT, epochs=28)
    texture = train_texture(train, val, OUT)
    choose(val, head, texture, OUT)
