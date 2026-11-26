"""Inference-compatible DINOv2 ViT-S/14 for the local PyTorch 1.12 runtime."""

import math

import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 384
PATCH = 14


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        batch, tokens, _ = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, 6, 64).permute(2, 0, 3, 1, 4)
        weights = ((q * .125) @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((weights @ v).transpose(1, 2).reshape(batch, tokens, WIDTH))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(WIDTH, WIDTH * 4)
        self.fc2 = nn.Linear(WIDTH * 4, WIDTH)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(WIDTH))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale()
        self.norm2 = nn.LayerNorm(WIDTH, eps=1e-6)
        self.mlp = FeedForward()
        self.ls2 = LayerScale()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, WIDTH, PATCH, stride=PATCH)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, WIDTH))
        self.mask_token = nn.Parameter(torch.zeros(1, WIDTH))
        self.blocks = nn.ModuleList(Block() for _ in range(12))
        self.norm = nn.LayerNorm(WIDTH, eps=1e-6)

    def position(self, height, width):
        count = self.pos_embed.shape[1] - 1
        side = math.isqrt(count)
        grid = self.pos_embed[:, 1:].reshape(1, side, side, WIDTH).permute(0, 3, 1, 2)
        # Match the original DINOv2 position interpolation offset.
        grid = F.interpolate(grid.float(), scale_factor=((height + .1) / side, (width + .1) / side),
                             mode="bicubic", align_corners=False)
        return torch.cat((self.pos_embed[:, :1], grid.flatten(2).transpose(1, 2)), dim=1)

    def forward(self, x):
        x = self.patch_embed.proj(x)
        height, width = x.shape[-2:]
        patch_tokens = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.to(x.dtype).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, patch_tokens), dim=1) + self.position(height, width).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def load_dino(weights, device):
    model = DinoSmall()
    source = torch.load(weights, map_location="cpu")
    model.load_state_dict(source, strict=True)
    return model.to(device).eval().requires_grad_(False)
