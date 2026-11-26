"""Read the earthquake damage regions from the MEDBFS Herat release."""

import argparse
from collections import Counter
import io
import os
from pathlib import Path, PurePosixPath
import zipfile

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np
import torch

from dino import load_dino
from features import embed, patch_pair, signal, STORE, WEIGHTS
from inventory import ROOT, write_json

DATA = ROOT / "data/external"
ARCHIVE_BYTES = 2286004809
RAW_START = 2036602908
RAW_END = 2285180544


class Parts(io.RawIOBase):
    def __init__(self):
        self.tail = (DATA / "afg_tail.bin").read_bytes()
        self.cut = ARCHIVE_BYTES - len(self.tail)
        raw = DATA / "afg_raw.bin"
        if raw.stat().st_size != RAW_END - RAW_START:
            raise ValueError("The selected raw archive range is incomplete")
        self.raw = raw.open("rb")
        self.offset = 0

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        self.offset = (0, self.offset, ARCHIVE_BYTES)[whence] + offset
        return self.offset

    def read(self, count=-1):
        if self.offset >= self.cut:
            i = self.offset - self.cut
            values = self.tail[i:] if count < 0 else self.tail[i:i + count]
        elif self.offset >= RAW_START:
            self.raw.seek(self.offset - RAW_START)
            values = self.raw.read(count)
        else:
            raise ValueError(f"Unrequested ZIP byte offset {self.offset}")
        self.offset += len(values)
        return values

    def close(self):
        self.raw.close()
        super().close()


def read_tiff(package, member, mask=False):
    raw = package.read(member)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to open {member.filename}")
    if mask:
        if image.ndim == 3:
            if not np.all(image[..., :3] == image[..., :1]):
                raise ValueError(f"Expected grayscale labels in {member.filename}")
            image = image[..., 0]
        return image
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    else:
        image = image[..., [2, 1, 0]]
    if image.dtype != np.uint8:
        sample = image[::5, ::5]
        lo, hi = np.quantile(sample.reshape(-1, 3), [.015, .985], axis=0)
        image = ((image.astype(np.float32) - lo) * 255 / np.maximum(hi - lo, 1)).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def lookup(z):
    pairs = {}
    for item in z.infolist():
        name = PurePosixPath(item.filename)
        if name.suffix.lower() not in (".tif", ".tiff"):
            continue
        kind = next((x for x in ("Pre-disaster", "Post-disaster", "Label") if x in name.parts), None)
        if kind is not None:
            pairs.setdefault(name.stem, {})[kind] = item
    complete = {key: value for key, value in pairs.items() if len(value) == 3}
    if len(complete) != 317:
        raise ValueError(f"Expected 317 paired images and masks, got {len(complete)}")
    return complete


def prepare(scale, inspect=False, cpu=False):
    device = torch.device("cuda" if not cpu and torch.cuda.is_available() else "cpu")
    encoder = None if inspect else load_dino(WEIGHTS, device)
    codes, groups = Counter(), Counter()
    features, numerical, ids = [], [], []
    with Parts() as source, zipfile.ZipFile(source) as z:
        pairs = lookup(z)
        for i, (key, items) in enumerate(sorted(pairs.items()), 1):
            mask = read_tiff(z, items["Label"], mask=True)
            values, counts = np.unique(mask, return_counts=True)
            codes.update(dict(zip(map(int, values), map(int, counts))))
            if not set(values).issubset({0, 1, 2}):
                raise ValueError(f"Unexpected class IDs {values} in {key}")
            bounds = []
            for category in (1, 2):
                _, _, stats, _ = cv2.connectedComponentsWithStats((mask == category).astype(np.uint8))
                for x, y, w, h, count in stats[1:]:
                    if min(w, h) >= 5 and max(w, h) <= 180 and count >= 30:
                        groups[category] += 1
                        bounds.append([x, y, x + w, y + h])
            if inspect or not bounds:
                continue
            before, after = [read_tiff(z, items[which]) for which in ("Pre-disaster", "Post-disaster")]
            if before.shape != after.shape or mask.shape != before.shape[:2]:
                raise ValueError(f"Pre/post and mask dimensions disagree: {key}")
            before, after = [np.repeat(cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None, fx=scale,
                                                fy=scale, interpolation=cv2.INTER_AREA)[..., None], 3, axis=2)
                             for im in (before, after)]
            patches, roi, texture = [], [], []
            for box in bounds:
                pair, region = patch_pair(before, after, np.asarray(box) * scale)
                patches.append(pair)
                roi.append(region)
                texture.append(signal(pair, region))
            for j in range(0, len(patches), 72):
                a, delta = embed(encoder, np.stack(patches[j:j + 72]), np.stack(roi[j:j + 72]), device)
                features.append(a.astype(np.float16))
                numerical.append(np.column_stack((np.stack(texture[j:j + 72]), delta)))
                ids.extend([key] * len(a))
            if i % 40 == 0:
                print(f"Herat earthquake regions: {i}/317, {len(ids)} retained", flush=True)
    audit = {"source": "https://zenodo.org/records/11241134", "tiles": 317,
             "pixel_codes": dict(codes), "retained_components": dict(groups),
             "notes": "Only grades 1 and 2 are used as positives. Both dates use intensity because "
                      "the pre-earthquake images are panchromatic.",
             "rescale": scale}
    write_json(DATA / "afghanistan_audit.json", audit)
    if not inspect:
        path = STORE / "afghanistan_severe.npz"
        np.savez(path, appearance=np.concatenate(features), change=np.concatenate(numerical),
                 grade=np.full(len(ids), 2, np.uint8), chip=np.asarray(ids))
    print(audit, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=.65)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    flags = parser.parse_args()
    if not .2 <= flags.scale <= 1:
        parser.error("scale must be between .2 and 1")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    prepare(flags.scale, flags.inspect, flags.cpu)
