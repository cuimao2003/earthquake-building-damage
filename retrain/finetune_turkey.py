"""Fine-tune the last ViT blocks on the reviewed Türkiye training groups."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

from classifier import PairHead, calibrate, report, matrix, predict_grade
from dino import dino_from_state, load_dino
from features import descriptors, patch_pair, read_rgb, signal, WEIGHTS
from inventory import ROOT, write_json
from turkey import DATA


class Patches:
    def __init__(self, name, size=112, store=None):
        if store is None:
            store = DATA / ("patches" if size == 112 else f"patches{size}")
        self.rgb = np.load(store / f"{name}_rgb.npy", mmap_mode="r")
        self.roi = np.load(store / f"{name}_roi.npy", mmap_mode="r")
        with np.load(store / f"{name}.npz") as numeric:
            self.change, self.grade = numeric["change"], numeric["grade"]
            self.chip = numeric["chip"]

    def batch(self, indices, device):
        rgb = np.ascontiguousarray(self.rgb[indices].transpose(0, 1, 4, 2, 3))
        arrays = (rgb, self.roi[indices], self.change[indices])
        values = tuple(torch.from_numpy(np.array(item, order="C", copy=not item.flags.writeable)).to(device)
                       for item in arrays)
        grade = getattr(self, "grade", None)
        return (*values, torch.from_numpy(np.array(grade[indices], copy=True)).to(device)
                if grade is not None else None)


def augmented(rgb, roi):
    rgb = rgb.float()
    if torch.rand(()) < .5:
        rgb, roi = rgb.flip(-1), roi.flip(-1)
    if torch.rand(()) < .40:
        rgb, roi = rgb.transpose(-1, -2), roi.transpose(-1, -2)
    if torch.rand(()) < .33:
        full = rgb.flatten(0, 1)
        resolution = int(torch.randint(int(rgb.shape[-1] * .32), int(rgb.shape[-1] * .84), (1,)))
        down = F.interpolate(full, size=(resolution, resolution), mode="area")
        rgb = F.interpolate(down, size=full.shape[-2:], mode="bilinear",
                            align_corners=False).reshape(rgb.shape)
    gain = torch.rand((rgb.size(0), 2, 3, 1, 1), device=rgb.device) * .30 + .85
    bias = torch.rand_like(gain) * 35 - 17.5
    return (gain * rgb + bias).clamp(0, 255), roi


def partial_collapse(rgb, roi, grade, rng):
    """Augment the intermediate grade with partial restoration of collapsed roofs."""
    candidates = torch.where(grade == 2)[0]
    count = min(len(candidates), max(1, round(len(candidates) * .18)))
    if count == 0:
        return rgb, grade, torch.ones_like(grade, dtype=torch.float32)
    chosen = candidates[torch.from_numpy(rng.choice(len(candidates), count, replace=False)).to(rgb.device)]
    before, after = rgb[chosen, 0].float(), rgb[chosen, 1].float()
    region = roi[chosen, None].float()
    context = 1 - region
    n = context.sum((2, 3), keepdim=True).clamp_min(1.)
    mu_pre = (before * context).sum((2, 3), keepdim=True) / n
    mu_post = (after * context).sum((2, 3), keepdim=True) / n
    adjusted = (before + mu_post - mu_pre).clamp(0, 255)
    ang = torch.as_tensor(rng.uniform(0, np.pi * 2, count), device=rgb.device, dtype=torch.float32)
    x = torch.linspace(-1, 1, rgb.size(-1), device=rgb.device)[None, None, None, :]
    y = torch.linspace(-1, 1, rgb.size(-2), device=rgb.device)[None, None, :, None]
    placement = torch.as_tensor(rng.uniform(-.17, .11, count), device=rgb.device, dtype=torch.float32)
    part = (x * ang.cos()[:, None, None, None] + y * ang.sin()[:, None, None, None]
            - placement[:, None, None, None])
    alpha = (.5 + part * 12).clamp(0., 1.) * region
    rgb = rgb.float()
    rgb[chosen, 1] = alpha * adjusted + (1 - alpha) * after
    grade = grade.clone()
    grade[chosen] = 1
    trust = torch.ones_like(grade, dtype=torch.float32)
    trust[chosen] = .45
    return rgb, grade, trust


def illumination_match(rgb, roi):
    paired = rgb.float()
    pre, post = paired[:, 0], paired[:, 1]
    ring = 1 - roi[:, None].float()
    area = ring.sum((2, 3), keepdim=True).clamp_min(1.)
    pre_mean = (pre * ring).sum((2, 3), keepdim=True) / area
    post_mean = (post * ring).sum((2, 3), keepdim=True) / area
    pre_var = ((pre - pre_mean).square() * ring).sum((2, 3), keepdim=True) / area
    post_var = ((post - post_mean).square() * ring).sum((2, 3), keepdim=True) / area
    gain = ((pre_var + 72) / (post_var + 72)).sqrt().clamp(.58, 1.65)
    corrected = ((post - post_mean) * gain + pre_mean).clamp(0, 255)
    return torch.stack((pre, (.65 * corrected + .35 * post)), dim=1)


def forward(encoder, head, r, m, x, local_evidence=False):
    a, d = descriptors(encoder, r, m, local_evidence)
    return head(a, torch.cat((x, d), dim=1))


@torch.no_grad()
def predict(encoder, head, data, device, batch=72, tta=False, local_evidence=False):
    head.eval()
    encoder.eval()
    output = []
    for i in range(0, len(data.rgb), batch):
        ids = slice(i, i + batch)
        rgb, mask, change, _ = data.batch(ids, device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            scores = forward(encoder, head, rgb, mask, change, local_evidence).softmax(dim=-1)
            if tta == "illumination":
                corrected = illumination_match(rgb, mask)
                scores += forward(encoder, head, corrected, mask, change, local_evidence).softmax(dim=-1)
                scores *= .5
            elif tta:
                scores += forward(encoder, head, rgb.flip(-1), mask.flip(-1), change,
                                  local_evidence).softmax(dim=-1)
                scores += forward(encoder, head, rgb.transpose(-1, -2), mask.transpose(-1, -2),
                                  change, local_evidence).softmax(dim=-1)
                scores /= 3
        output.append(scores.float().cpu().numpy())
    return np.concatenate(output)


def save(path, encoder, head, **metadata):
    checkpoint = {"encoder": encoder.state_dict(), "head": head.state_dict(),
                  "n_features": head.center.numel(), **metadata}
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load(path, device):
    state = torch.load(path, map_location="cpu")
    encoder = dino_from_state(state["encoder"], device)
    head = PairHead(state["n_features"], visual_width=encoder.width).to(device)
    head.load_state_dict(state["head"])
    return encoder, head.eval(), state


def train(args, device, store=None, sample_fractions=(.36, .27), checkpoint_name="turkey_vit.pt",
          auxiliary=None):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, validation = [Patches(split, args.patch_size, store) for split in ("train", "val")]
    if args.init is None:
        encoder = load_dino(getattr(args, "weights", WEIGHTS), device)
        head = PairHead(train.change.shape[1] + 3, visual_width=encoder.width).to(device)
    else:
        encoder, head, _ = load(args.init, device)
    for layer in encoder.blocks[-args.blocks:]:
        layer.requires_grad_(True)
    encoder.norm.requires_grad_(True)
    head.center[:-3].copy_(torch.from_numpy(train.change.mean(0)))
    head.scale[:-3].copy_(torch.from_numpy(train.change.std(0).clip(.015)))
    head.scale[-3:].fill_(.25)
    optim = torch.optim.AdamW([{"params": [p for p in encoder.parameters() if p.requires_grad],
                                "lr": args.lr},
                               {"params": head.parameters(), "lr": 4.0 * args.lr}], weight_decay=.035)
    if auxiliary is not None:
        auxiliary_head = PairHead(auxiliary.change.shape[1] + 3, visual_width=encoder.width).to(device)
        auxiliary_head.center[:-3].copy_(torch.from_numpy(auxiliary.change.mean(0)))
        auxiliary_head.scale[:-3].copy_(torch.from_numpy(auxiliary.change.std(0).clip(.015)))
        auxiliary_head.scale[-3:].fill_(.25)
        aux_groups = [np.flatnonzero(auxiliary.grade == grade) for grade in np.unique(auxiliary.grade)]
        optim.add_param_group({"params": auxiliary_head.parameters(), "lr": 4.0 * args.lr})
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=.00001)
    groups = [np.flatnonzero(train.grade == k) for k in range(3)]
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best = -1.
    history = []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        head.train()
        sums = torch.zeros((), device=device)
        for _ in range(args.steps):
            sizes = [int(fraction * args.batch) for fraction in sample_fractions]
            sizes.append(args.batch - sum(sizes))
            ids = np.concatenate([rng.choice(g, size, replace=True) for g, size in zip(groups, sizes)])
            rgb, roi, change, grade = train.batch(ids, device)
            if args.partial_augment:
                rgb, grade, trust = partial_collapse(rgb, roi, grade, rng)
                change[trust < 1.] = head.center[:-3]
            else:
                trust = torch.ones_like(grade, dtype=torch.float32)
            rgb, roi = augmented(rgb, roi)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                z = forward(encoder, head, rgb, roi, change, args.local_evidence).float()
                classification = F.cross_entropy(z, grade.long(), label_smoothing=.045, reduction="none")
                collapse = F.binary_cross_entropy_with_logits(
                    z[:, 2] - z[:, :2].logsumexp(1), (grade == 2).float(), reduction="none")
                loss = ((classification + .16 * collapse) * trust).sum() / trust.sum()
                if auxiliary is not None:
                    extra = np.concatenate([rng.choice(group, 6, replace=True) for group in aux_groups])
                    views, region, numeric, damage = auxiliary.batch(extra, device)
                    views, region = augmented(views, region)
                    graded = forward(encoder, auxiliary_head, views, region, numeric, args.local_evidence)
                    loss = loss + .38 * F.cross_entropy(graded.float(), damage.long(), label_smoothing=.045)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(optim.param_groups[0]["params"], 5.)
            scaler.step(optim)
            scaler.update()
            sums += loss.detach()
        schedule.step()
        row = {"epoch": epoch, "loss": float(sums) / args.steps, "elapsed": time.monotonic() - start}
        if epoch == 1 or epoch % 2 == 0 or epoch == args.epochs:
            scores = predict(encoder, head, validation, device, batch=max(args.batch, 32),
                             local_evidence=args.local_evidence)
            thresholds, performance = calibrate(scores, validation.grade)
            row.update({"thresholds": thresholds, "validation": performance})
            if performance["macro_f1"] > best:
                best = performance["macro_f1"]
                save(args.out / checkpoint_name, encoder, head, patch_size=args.patch_size,
                     local_evidence=args.local_evidence, partial_augment=args.partial_augment, **row)
            print(f"{epoch}: loss {row['loss']:.3f}, validation F1 {performance['macro_f1']:.4f}, "
                  f"grades {[round(item['f1'], 3) for item in performance['classes'].values()]} "
                  f"time {row['elapsed']:.1f}s", flush=True)
        history.append(row)
        write_json(args.out / "history.json", history)


def saved_scores(checkpoint, split, device, tta):
    encoder, head, state = load(checkpoint, device)
    size = state.get("patch_size", 112)
    data = Patches(split, size)
    scores = predict(encoder, head, data, device, batch=40 if size > 112 else 72,
                     tta=tta, local_evidence=state.get("local_evidence", False))
    return data, scores, state["epoch"]


def validation_row(data, scores, members):
    cuts, performance = calibrate(scores, data.grade)
    return {"models": members, "thresholds": cuts, "validation": performance}


def calibration_candidates(checkpoint, device):
    encoder, head, state = load(checkpoint, device)
    size = state.get("patch_size", 112)
    validation = Patches("val", size)
    evaluated = []
    for tta in (False, True, "illumination"):
        scores = predict(encoder, head, validation, device, batch=40 if size > 112 else 72,
                         tta=tta, local_evidence=state.get("local_evidence", False))
        member = {"path": str(checkpoint.resolve()), "tta": tta, "weight": 1., "epoch": state["epoch"]}
        result = validation_row(validation, scores, [member])
        evaluated.append((result, scores))
    return validation, evaluated


def test_examples(data, labels, output):
    rng = np.random.default_rng(77051)
    selected = np.concatenate([rng.choice(np.flatnonzero(data.grade == k), 3, replace=False)
                               for k in range(3)])
    page = Image.new("RGB", (812, 462), "#13202b")
    draw = ImageDraw.Draw(page)
    for i, index in enumerate(selected):
        x, y = 10 + (i % 3) * 267, 6 + (i // 3) * 153
        before, after = [Image.fromarray(im).resize((127, 127)) for im in data.rgb[index]]
        page.paste(before, (x, y + 23))
        page.paste(after, (x + 130, y + 23))
        color = "#aee5c4" if labels[index] == data.grade[index] else "#ffab9a"
        draw.text((x, y + 5), f"Scene {data.chip[index]} · Human {'ABC'[data.grade[index]]} / "
                  f"Model {'ABC'[labels[index]]}", fill=color)
    page.save(output)


def calibrate_final(args, device):
    primary = args.out / "turkey_vit.pt"
    validation, evaluated = calibration_candidates(primary, device)
    candidates = [row for row, _ in evaluated]
    if args.combine:
        other, extra = calibration_candidates(args.combine, device)
        if not np.array_equal(validation.grade, other.grade) or not np.array_equal(validation.chip, other.chip):
            raise ValueError("The checkpoints do not share the same validation buildings")
        candidates += [row for row, _ in extra]
        criterion = lambda item: item[0]["validation"]["macro_f1"]
        a, p = max(evaluated, key=criterion)
        b, q = max(extra, key=criterion)
        for mix in (.25, .50, .75):
            members = [{**a["models"][0], "weight": 1 - mix}, {**b["models"][0], "weight": mix}]
            candidates.append(validation_row(validation, (1 - mix) * p + mix * q, members))
    best = max(candidates, key=lambda row: row["validation"]["macro_f1"])
    write_json(args.out / "selection.json", {"selected": best, "candidates": candidates})
    print(json.dumps({"validation_macro_f1": best["validation"]["macro_f1"],
                      "members": best["models"]}), flush=True)


def test_final(args, device):
    best = json.loads((args.out / "selection.json").read_bytes())["selected"]
    test, combined = None, None
    for member in best["models"]:
        current, scores, _ = saved_scores(Path(member["path"]), "test", device, member["tta"])
        if test is None:
            test, combined = current, scores * member["weight"]
        else:
            if not np.array_equal(test.grade, current.grade) or not np.array_equal(test.chip, current.chip):
                raise ValueError("The checkpoints do not share the same test buildings")
            combined += scores * member["weight"]
    labels = predict_grade(combined, np.asarray(best["thresholds"]))
    results = report(matrix(test.grade, labels))
    results["grade_names"] = ["A", "B", "C"]
    results["test_scenes"] = sorted(set(test.chip.tolist()))
    results["settings"] = best
    write_json(args.out / "test.json", results)
    test_examples(test, labels, args.out / "test_examples.png")
    print(json.dumps(results, indent=2), flush=True)


class ImagePatches(Patches):
    def __init__(self, pre, post, boxes, size):
        height, width = pre.shape[:2]
        self.rgb = []
        self.roi = []
        self.change = []
        for bounds in boxes:
            x0, y0, x1, y1 = bounds
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"Building box is outside the image: {bounds}")
            image, mask = patch_pair(pre, post, bounds, size=size)
            self.rgb.append(image)
            self.roi.append(mask)
            self.change.append(signal(image, mask))
        self.rgb, self.roi, self.change = map(np.stack, (self.rgb, self.roi, self.change))


def grade_image(args, device):
    settings = json.loads((args.out / "selection.json").read_bytes())["selected"]
    before, after = read_rgb(args.pre), read_rgb(args.post)
    if before.shape != after.shape:
        raise ValueError("Pre/post images must share the same pixel frame and dimensions")
    boxes = np.asarray(json.loads(args.boxes.read_bytes()), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes) or not np.isfinite(boxes).all():
        raise ValueError("boxes must be a JSON list of [left, top, right, bottom] arrays")
    combined = np.zeros((len(boxes), 3), np.float32)
    for member in settings["models"]:
        encoder, head, state = load(Path(member["path"]), device)
        size = state.get("patch_size", 112)
        samples = ImagePatches(before, after, boxes, size)
        combined += member["weight"] * predict(encoder, head, samples, device,
                                                 batch=40 if size > 112 else 72,
                                                 tta=member["tta"],
                                                 local_evidence=state.get("local_evidence", False))
    grades = predict_grade(combined, np.asarray(settings["thresholds"]))
    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    palette = ("#52b3e8", "#ffc231", "#ff4855")
    overlaid = Image.fromarray(after.copy())
    draw = ImageDraw.Draw(overlaid)
    with (args.prediction_dir / "grades.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["left", "top", "right", "bottom", "grade", "score_A", "score_B", "score_C"])
        for box, grade, scores in zip(boxes, grades, combined):
            coords = box.astype(float).tolist()
            writer.writerow([*coords, "ABC"[grade], *scores.tolist()])
            draw.rectangle(coords, outline=palette[grade], width=3)
    overlaid.save(args.prediction_dir / "grades_overlay.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run/turkey_vit")
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=36)
    parser.add_argument("--blocks", type=int, default=3, choices=(1, 2, 3, 4, 5, 6))
    parser.add_argument("--lr", type=float, default=.00006)
    parser.add_argument("--seed", type=int, default=93742)
    parser.add_argument("--patch-size", type=int, default=112, choices=(112, 168))
    parser.add_argument("--local-evidence", action="store_true")
    parser.add_argument("--partial-augment", action="store_true")
    parser.add_argument("--init", type=Path)
    parser.add_argument("--combine", type=Path, help="Compare a second checkpoint on validation")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--pre", type=Path)
    parser.add_argument("--post", type=Path)
    parser.add_argument("--boxes", type=Path)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch) < 3 or args.lr <= 0:
        parser.error("invalid training parameters")
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if not args.cpu and torch.cuda.is_available() else "cpu")
    if args.pre or args.post or args.boxes or args.prediction_dir:
        if not all((args.pre, args.post, args.boxes, args.prediction_dir)):
            parser.error("image inference requires --pre, --post, --boxes and --prediction-dir")
        grade_image(args, device)
    elif args.calibrate:
        calibrate_final(args, device)
    elif args.test:
        test_final(args, device)
    else:
        train(args, device)
