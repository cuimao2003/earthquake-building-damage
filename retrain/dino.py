"""DINOv2 ViT-S/B/L backbones for the local PyTorch 1.12 runtime."""

import math

import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 384
PATCH = 14


class Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.heads = width // 64
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, 64).permute(2, 0, 3, 1, 4)
        weights = ((q * .125) @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((weights @ v).transpose(1, 2).reshape(batch, tokens, width))


class FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = Attention(width)
        self.ls1 = LayerScale(width)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = FeedForward(width)
        self.ls2 = LayerScale(width)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoVision(nn.Module):
    def __init__(self, width, depth, positions):
        super().__init__()
        self.width = width
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, width, PATCH, stride=PATCH)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, positions, width))
        self.mask_token = nn.Parameter(torch.zeros(1, width))
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)

    def position(self, height, width):
        count = self.pos_embed.shape[1] - 1
        side = math.isqrt(count)
        grid = self.pos_embed[:, 1:].reshape(1, side, side, self.width).permute(0, 3, 1, 2)
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


def dino_from_state(source, device):
    _, positions, width = source["pos_embed"].shape
    depths = {384: 12, 768: 12, 1024: 24}
    if width not in depths or math.isqrt(positions - 1) ** 2 != positions - 1:
        raise ValueError("Unsupported DINOv2 backbone")
    model = DinoVision(width, depths[width], positions)
    model.load_state_dict(source, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_dino(weights, device):
    return dino_from_state(torch.load(weights, map_location="cpu"), device)
