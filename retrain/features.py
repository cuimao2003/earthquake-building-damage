"""Extract paired Transformer embeddings and local roof change descriptors."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from shapely import wkt

from dino import PATCH, load_dino
from inventory import LABELS, ROOT

SIZE = 112
WEIGHTS = ROOT / "weights/dinov2_vits14_pretrain.pth"
STORE = ROOT / "data/features"
ALIGNED = ROOT / "data/aligned_keypoint"


def read_rgb(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image {path}")
    return np.ascontiguousarray(image[:, :, ::-1])


def align(pre, post, method="phase"):
    if pre.shape != post.shape:
        raise ValueError("Pre and post image sizes do not match")
    if method == "keypoint":
        refined = keypoint_align(pre, post)
        if refined is not None:
            return refined
    scores = []
    for img in (pre, post):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        residual = gray - cv2.GaussianBlur(gray, (0, 0), 3.0)
        scores.append(residual / max(residual.std(), 1.))
    h, w = scores[0].shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(*scores, window)
    if not np.isfinite([dx, dy, response]).all() or response < .14 or max(abs(dx), abs(dy)) > 12:
        return post, (0., 0., float(response))
    transform = np.float32([[1, 0, -dx], [0, 1, -dy]])
    result = cv2.warpAffine(post, transform, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return result, (float(dx), float(dy), float(response))


def keypoint_align(pre, post):
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
    extractor = cv2.SIFT_create(nfeatures=2200, contrastThreshold=.018)
    points, features = [], []
    for img in (pre, post):
        gray = clahe.apply(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        keys, values = extractor.detectAndCompute(gray, None)
        if values is None or len(values) < 20:
            return None
        points.append(np.float32([k.pt for k in keys]))
        features.append(np.sqrt(values / values.sum(1, keepdims=True).clip(1e-4)))
    near = (np.abs(points[0][:, None, 0] - points[1][None, :, 0]) <= 85)
    near &= (np.abs(points[0][:, None, 1] - points[1][None, :, 1]) <= 85)
    matching = cv2.BFMatcher(cv2.NORM_L2).knnMatch(features[0], features[1], k=2,
                                                  mask=near.astype(np.uint8))
    pairs = [m[0] for m in matching if len(m) == 2 and m[0].distance < .76 * m[1].distance]
    if len(pairs) < 16:
        return None
    p = np.float32([points[0][m.queryIdx] for m in pairs])
    q = np.float32([points[1][m.trainIdx] for m in pairs])
    transform, status = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
                                                    ransacReprojThreshold=3.5, maxIters=1800)
    if transform is None:
        return None
    inlier = status.ravel().astype(bool)
    if (inlier.sum() < 14 or inlier.mean() < .21 or np.ptp(p[inlier], axis=0).min() < 120
            or not .96 <= np.linalg.norm(transform[:, 0]) <= 1.04 or abs(transform[0, 1]) > .045):
        return None
    center = np.asarray([pre.shape[1] / 2, pre.shape[0] / 2])
    dx, dy = transform[:, :2] @ center + transform[:, 2] - center
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    warped = cv2.warpAffine(post, transform, (pre.shape[1], pre.shape[0]), flags=flags,
                             borderMode=cv2.BORDER_REFLECT_101)
    return warped, (float(dx), float(dy), float(inlier.mean()))


def crop_window(origin, extent, size):
    start = int(np.floor(origin)) - 2
    stop = int(np.ceil(origin + extent)) + 3
    return max(0, min(start, 2 * size - stop - 2)), min(size, max(stop, -start + 2))


def patch_pair(pre, post, bounds, extent_factor=1.8, size=SIZE):
    x0, y0, x1, y1 = bounds
    width, height = max(1., x1 - x0), max(1., y1 - y0)
    extent = float(np.clip(max(width, height) * extent_factor, 24, 220))
    ox, oy = (x0 + x1 - extent) / 2, (y0 + y1 - extent) / 2
    scale = size / extent
    left, right = crop_window(ox, extent, pre.shape[1])
    top, bottom = crop_window(oy, extent, pre.shape[0])
    map_to_patch = np.float32([[scale, 0, scale * (left - ox)], [0, scale, scale * (top - oy)]])
    images = np.stack([cv2.warpAffine(np.ascontiguousarray(im[top:bottom, left:right]),
                                       map_to_patch, (size, size), borderMode=cv2.BORDER_REFLECT_101)
                       for im in (pre, post)])
    y, x = np.ogrid[:size, :size]
    rx, ry = np.clip([width * scale * .50, height * scale * .50], 1.5, size * .48)
    region = ((x - size / 2) / rx) ** 2 + ((y - size / 2) / ry) ** 2 <= 1.
    return images, region


def moments(values):
    return np.concatenate((values.mean(0), values.std(0), np.quantile(values, [.15, .85], axis=0).ravel()))


def texture(gray, mask):
    quantized = (gray.clip(0, 255).astype(np.int32) // 16)
    hist = np.zeros(256, np.float64)
    for distance in (1, 3):
        for a, b, keep in ((quantized[:, :-distance], quantized[:, distance:],
                            mask[:, :-distance] & mask[:, distance:]),
                           (quantized[:-distance], quantized[distance:],
                            mask[:-distance] & mask[distance:])):
            hist += np.bincount(16 * a[keep] + b[keep], minlength=256)
    prob = hist / max(1., hist.sum())
    indices = np.arange(256)
    dist = (indices // 16 - indices % 16) ** 2
    return np.array([prob @ dist / 225, prob @ (1. / (1. + dist)), prob @ prob,
                     -(prob * np.log(prob.clip(1e-9))).sum() / 6.])


def signal(images, region):
    dates = []
    grayscale = []
    for im in images:
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grayscale.append(gray)
        blurred = cv2.GaussianBlur(gray, (0, 0), 2)
        detail = np.abs(gray - blurred) / 255.
        x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        edge = np.hypot(x, y) / 1020.
        values = np.dstack((im.astype(np.float32) / 255, detail, edge))
        interior = moments(values[region])
        context = moments(values[~region])
        dates.append(np.concatenate((interior, interior - context, texture(gray, region))))
    pre, post = grayscale
    a, b = [g - cv2.GaussianBlur(g, (0, 0), 3) for g in grayscale]
    similarity = []
    for mask in (region, ~region):
        av, bv = a[mask], b[mask]
        denom = np.linalg.norm(av) * np.linalg.norm(bv)
        similarity.extend([float(av @ bv / max(denom, 1.)),
                           float(np.abs(av - bv).mean() / 255.),
                           float(np.abs(pre[mask] - post[mask]).mean() / 255.)])
    return np.concatenate((dates[0], dates[1], dates[1] - dates[0], similarity)).astype(np.float32)


def descriptors(encoder, batch, regions, local_evidence=False):
    n = batch.shape[0]
    batch = batch.flatten(0, 1).float().div(255.)
    mean = batch.new_tensor([.485, .456, .406])[None, :, None, None]
    sd = batch.new_tensor([.229, .224, .225])[None, :, None, None]
    batch = (batch - mean) / sd
    tokens = encoder(batch).float()
    cls = F.normalize(tokens[:, 0], dim=1).reshape(n, 2, -1)
    patch = F.normalize(tokens[:, 1:], dim=-1).reshape(n, 2, -1, cls.shape[-1])
    grid = F.avg_pool2d(regions[:, None].float(), PATCH)
    weights = grid.flatten(1)
    total = weights.sum(1, keepdim=True).clamp_min(1e-4)
    same = (patch[:, 0] * patch[:, 1]).sum(-1)
    outside = 1 - weights
    if local_evidence:
        # Permit a one-token displacement before ranking visual changes in the roof.
        with torch.no_grad():
            height, width = grid.shape[-2:]
            channels = patch.shape[-1]
            before = patch[:, 0].reshape(n, height, width, channels).permute(0, 3, 1, 2)
            after = patch[:, 1].reshape(n, height, width, channels).permute(0, 3, 1, 2)
            padded = F.pad(before, (1, 1, 1, 1), mode="replicate")
            best = same.reshape(n, height, width)
            for y in range(3):
                for x in range(3):
                    nearby = (padded[:, :, y:y + height, x:x + width] * after).sum(dim=1)
                    best = torch.maximum(best, nearby)
            same = .5 * same + .5 * best.flatten(1)
            distance = (1 - same).clamp_min(.01)
            context = (distance * outside).sum(1, keepdim=True) / outside.sum(1, keepdim=True).clamp_min(1e-4)
            emphasis = weights * (1 + (distance / context.clamp_min(.01)).clamp(max=4.))
    else:
        emphasis = weights
    pooled = F.normalize((patch * emphasis[:, None, :, None]).sum(2), dim=-1)
    similarities = torch.cat(((same * weights).sum(1, keepdim=True) / total,
                              (same * outside).sum(1, keepdim=True) /
                              outside.sum(1, keepdim=True).clamp_min(1e-4),
                              (cls[:, 0] * cls[:, 1]).sum(1, keepdim=True)), dim=1)
    return torch.cat((cls, pooled), dim=1), similarities


@torch.no_grad()
def embed(encoder, views, regions, device, local_evidence=False):
    batch = torch.from_numpy(np.ascontiguousarray(views.transpose(0, 1, 4, 2, 3))).to(device)
    roi = torch.from_numpy(np.ascontiguousarray(regions)).to(device)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        appearance, numerical = descriptors(encoder, batch, roi, local_evidence)
    return appearance.cpu().numpy(), numerical.cpu().numpy()


def jitter_bounds(bounds, rng, shape):
    low = np.asarray(bounds[:2])
    high = np.asarray(bounds[2:])
    extent = high - low
    center = (low + high) * .5 + extent * rng.uniform(-.16, .16, size=2)
    half = extent * rng.uniform(.81, 1.20, size=2) * .5
    cap = np.asarray(shape[1::-1])
    return (*np.clip(center - half, 0, cap), *np.clip(center + half, 0, cap))


def read_record(record, negative_limit, registration_method, jitter=False, size=SIZE):
    path = Path(record["source"])
    chip = record["chip"]
    pre = read_rgb(path / "images" / f"{chip}_pre_disaster.png")
    aligned = ALIGNED / f"{chip}.npz"
    if registration_method == "keypoint" and aligned.is_file():
        with np.load(aligned) as archive:
            post, registration = archive["post"], archive["registration"].tolist()
    else:
        post = read_rgb(path / "images" / f"{chip}_post_disaster.png")
        post, registration = align(pre, post, registration_method)
    raw = json.loads((path / "labels" / f"{chip}_post_disaster.json").read_bytes())["features"]["xy"]
    known = [item for item in raw if item["properties"].get("subtype") in LABELS]
    if negative_limit:
        intact = [i for i, b in enumerate(known) if b["properties"]["subtype"] == "no-damage"]
        if len(intact) > negative_limit:
            rng = np.random.default_rng(sum(map(ord, chip)) * 347)
            exclude = set(rng.permutation(intact)[negative_limit:].tolist())
            known = [item for i, item in enumerate(known) if i not in exclude]
    patches, masks, statistics, labels, ids = [], [], [], [], []
    jitter_rng = np.random.default_rng(sum(map(ord, chip)) * 1093) if jitter else None
    for item in known:
        bounds = wkt.loads(item["wkt"]).bounds
        if len(bounds) != 4:
            raise ValueError(f"Invalid building {item['properties']['uid']}")
        grade = LABELS[item["properties"]["subtype"]]
        repeats = 3 if jitter and grade else 1
        for i in range(repeats):
            crop = jitter_bounds(bounds, jitter_rng, pre.shape) if jitter else bounds
            img, mask = patch_pair(pre, post, crop, size=size)
            patches.append(img)
            masks.append(mask)
            statistics.append(signal(img, mask))
            labels.append(grade)
            uid = item["properties"]["uid"]
            ids.append(f"{uid}/jitter-{i}" if jitter else uid)
    return patches, masks, statistics, labels, ids, registration


def cache_suffix(registration, transferred=False):
    return ("_keypoint" if registration == "keypoint" else "") + ("_transferred" if transferred else "")


def compute_features(encoder, records, split, device, batch, limit, registration="phase",
                     transferred=False, jitter=False):
    suffix = cache_suffix(registration, transferred) + ("_jittered" if jitter else "")
    output = STORE / f"{split}{suffix}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    image_tokens, metadata, grades, scenes, uids = [], [], [], [], []
    offsets = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0, len(records), 4):
            group = records[offset:offset + 4]
            results = pool.map(read_record, group, [limit] * len(group), [registration] * len(group),
                               [jitter] * len(group))
            for current, (images, roi, numeric, labels, ids, registration) in zip(group, results):
                offsets.append({"chip": current["chip"], "align": registration})
                for start in range(0, len(labels), batch):
                    stop = min(len(labels), start + batch)
                    vectors, similarity = embed(encoder, np.stack(images[start:stop]),
                                                np.stack(roi[start:stop]), device)
                    image_tokens.append(vectors.astype(np.float16))
                    metadata.append(np.concatenate((np.stack(numeric[start:stop]), similarity), axis=1))
                grades.extend(labels)
                scenes.extend([current["chip"]] * len(labels))
                uids.extend(ids)
            if offset % 20 == 0 or offset + 4 >= len(records):
                print(f"{split}: tiles {min(offset + 4, len(records))}/{len(records)}, "
                      f"buildings {len(grades):,}", flush=True)
    temp = output.with_name(output.stem + ".tmp.npz")
    np.savez(temp, appearance=np.concatenate(image_tokens), change=np.concatenate(metadata),
             grade=np.asarray(grades, np.uint8), chip=np.asarray(scenes), uid=np.asarray(uids))
    temp.replace(output)
    (STORE / f"{split}{suffix}_registration.json").write_text(json.dumps(offsets, indent=2), encoding="utf-8")
    print(f"{split}: {dict(Counter(grades))}", flush=True)


def cache_alignment(row):
    chip = row["chip"]
    result = ALIGNED / f"{chip}.npz"
    if result.is_file():
        return
    source = Path(row["source"]) / "images"
    before, after = [read_rgb(source / f"{chip}_{when}_disaster.png") for when in ("pre", "post")]
    aligned, parameters = align(before, after, "keypoint")
    temporary = result.with_suffix(".tmp.npz")
    np.savez(temporary, post=aligned, registration=parameters)
    temporary.replace(result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("split", nargs="+", choices=("train", "val", "test"))
    cli.add_argument("--batch", type=int, default=72)
    cli.add_argument("--train-intact-per-tile", type=int, default=72)
    cli.add_argument("--registration", choices=("phase", "keypoint"), default="phase")
    cli.add_argument("--cache-alignment", action="store_true")
    cli.add_argument("--encoder-checkpoint", type=Path, help="A fine-tuned Türkiye encoder checkpoint")
    cli.add_argument("--jitter-train", action="store_true", help="Cache offset boxes derived from training labels")
    args = cli.parse_args()
    if args.batch < 1 or args.train_intact_per_tile < 1:
        cli.error("batch and intact sample limit must be positive")
    if args.jitter_train and (args.split != ["train"] or args.cache_alignment):
        cli.error("jitter-train requires the single train split and feature extraction")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if args.cache_alignment:
        ALIGNED.mkdir(exist_ok=True)
        rows = [row for split in args.split for row in manifest[split]]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i, _ in enumerate(pool.map(cache_alignment, rows), 1):
                if i % 20 == 0 or i == len(rows):
                    print(f"Keypoint registration: {i}/{len(rows)}", flush=True)
        raise SystemExit(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_dino(WEIGHTS, device)
    if args.encoder_checkpoint:
        weights = torch.load(args.encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(weights["encoder"], strict=True)
    for split in args.split:
        compute_features(encoder, manifest[split], split, device, args.batch,
                         args.train_intact_per_tile if split == "train" else None,
                         args.registration, args.encoder_checkpoint is not None, args.jitter_train)
