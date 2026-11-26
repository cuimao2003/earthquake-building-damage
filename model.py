"""Compact Transformer for building extraction and paired-image damage mapping."""

import torch
from torch import nn
from torch.nn import functional as F


def conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class DamageTransformer(nn.Module):
    channels = (16, 24, 40, 64)

    def __init__(self):
        super().__init__()
        channels = self.channels
        self.pyramid = nn.ModuleList(conv(source, dest, stride=2)
                                     for source, dest in zip((3, *channels), channels))
        width = channels[-1]
        self.position = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.attention = nn.ModuleList(
            nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=128, dropout=0.1,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(2)
        )
        self.token_norm = nn.LayerNorm(width)
        self.fusion = nn.ModuleList(nn.Conv2d(3 * ch, ch, 1) for ch in channels)
        for layer in self.fusion:
            nn.init.normal_(layer.weight, std=0.008)
            nn.init.zeros_(layer.bias)
        self.decoder = nn.ModuleList(conv(a + b, b) for a, b in
                                     zip(channels[:0:-1], channels[-2::-1]))
        self.footprint = nn.Conv2d(channels[0], 1, 1)
        self.damage_context = conv(width, channels[0])
        self.damage = nn.Sequential(conv(2 * channels[0], 24), nn.Conv2d(24, 3, 1))

    def encode(self, x):
        skips = []
        for layer in self.pyramid:
            x = layer(x)
            skips.append(x)
        x = x + self.position(x)
        tokens = x.flatten(2).transpose(1, 2)
        for layer in self.attention:
            tokens = layer(tokens)
        skips[-1] = self.token_norm(tokens).transpose(1, 2).reshape(x.shape)
        return skips

    def decode(self, features):
        x = features[-1]
        for layer, skip in zip(self.decoder, reversed(features[:-1])):
            x = layer(torch.cat((resize(x, skip.shape[-2:]), skip), dim=1))
        return x

    def forward(self, pre, post=None):
        shape = pre.shape[-2:]
        if post is None:
            return resize(self.footprint(self.decode(self.encode(pre))), shape)
        features = []
        for layer, feature in zip(self.fusion, self.encode(torch.cat((pre, post)))):
            before, after = feature.chunk(2)
            features.append(before + layer(torch.cat((before, after, after - before), dim=1)))
        local = self.decode(features)
        building = resize(self.footprint(local), shape)
        context = resize(self.damage_context(features[-1]), local.shape[-2:])
        damage = resize(self.damage(torch.cat((local, context), dim=1)), shape)
        return building, damage

    def probabilities(self, pre, post):
        footprint, damage = self(pre, post)
        return torch.cat((footprint.float().sigmoid(), damage.float().softmax(dim=1)), dim=1)
