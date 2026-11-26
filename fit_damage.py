"""Fine-tune the shared Transformer on earthquake pairs."""

from functools import lru_cache
import json
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from evaluation import validation
from prepare import CLASSES, SIZE, write_json
from train import (augment_image, footprint_loss, image_tensor, load_checkpoint,
                   save_checkpoint, tensor_from_batch, transform_chip)


class EarthquakeCrops:
    def __init__(self, cache, records, seed, focused, batch_size):
        self.tiles = [cache / "earthquake" / record["file"] for record in records]
        self.rng = np.random.default_rng(seed)
        self.focused = focused
        self.buffer = np.empty((batch_size, SIZE, SIZE, 7), np.uint8)
        self.damaged = {2: [], 3: []}
        for index, record in enumerate(records):
            for building in record["buildings"]:
                if building["class_id"] in self.damaged:
                    self.damaged[building["class_id"]].append((index, *building["center"]))
        if not all(self.damaged.values()):
            raise ValueError("Training must contain annotations for both damage grades")

    @lru_cache(maxsize=12)
    def tile(self, index):
        return np.load(self.tiles[index], mmap_mode="r")

    def crop(self):
        rng = self.rng
        mode = rng.choice((0, 2, 3), p=(0.45, 0.32, 0.23))
        side = SIZE
        if mode:
            options = self.damaged[mode]
            index, center_x, center_y = options[rng.integers(len(options))]
            chip = self.tile(index)
            if self.focused and rng.random() < 0.65:
                side = int(rng.integers(144, SIZE + 1))
            jitter = side // 3 if self.focused else 88
            x = int(np.clip(center_x - side // 2 + rng.integers(-jitter, jitter + 1),
                            0, chip.shape[1] - side))
            y = int(np.clip(center_y - side // 2 + rng.integers(-jitter, jitter + 1),
                            0, chip.shape[0] - side))
        else:
            chip = self.tile(int(rng.integers(len(self.tiles))))
            y = rng.integers(chip.shape[0] - SIZE + 1)
            x = rng.integers(chip.shape[1] - SIZE + 1)
        array = chip[y:y + side, x:x + side]
        if side != SIZE:
            rgb = cv2.resize(np.ascontiguousarray(array[..., :6]), (SIZE, SIZE),
                             interpolation=cv2.INTER_LINEAR)
            classes = cv2.resize(np.ascontiguousarray(array[..., 6]), (SIZE, SIZE),
                                 interpolation=cv2.INTER_NEAREST)
            array = np.dstack((rgb, classes))
        return transform_chip(array, rng)

    def batch(self):
        for target in self.buffer:
            target[:] = self.crop()
        return self.buffer


def damage_loss(logits, target):
    # Unknown buildings contribute to the footprint head only.
    known = (target > 0) & (target < 4)
    target = torch.where(known, target - 1, 255).long()
    losses = F.cross_entropy(logits.float(), target, ignore_index=255, reduction="none",
                             label_smoothing=0.015)
    masks = target[:, None] == torch.arange(3, device=target.device)[None, :, None, None]
    count = masks.sum(dim=(0, 2, 3)).float()
    per_class = (losses[:, None] * masks).sum(dim=(0, 2, 3)) / count.clamp_min(1.)
    weighted = per_class.sum() / (count > 0).sum().clamp_min(1)
    mean = losses.sum() / count.sum().clamp_min(1.)
    return 0.55 * mean + 0.45 * weighted


def hierarchical_damage_loss(logits, labels):
    logits = logits.float()
    aggregate = logits[:, 1:].logsumexp(dim=1) - logits[:, 0]
    detection_loss = F.softplus(aggregate)
    damage_loss = F.softplus(-aggregate)
    intact = labels == 1
    damaged = (labels == 2) | (labels == 3)
    severity = logits[:, 2] - logits[:, 1]
    minor = labels == 2
    severe = labels == 3

    def masked_average(loss, mask):
        count = mask.sum(dim=(1, 2))
        per_crop = (loss * mask).sum(dim=(1, 2)) / count.clamp_min(1)
        return per_crop.sum() / (count > 0).sum().clamp_min(1)

    distinguish_damage = 0.52 * masked_average(detection_loss, intact)
    distinguish_damage += 0.48 * masked_average(damage_loss, damaged)
    distinguish_grade = 0.5 * (masked_average(F.softplus(severity), minor)
                               + masked_average(F.softplus(-severity), severe))
    return distinguish_damage + 0.38 * distinguish_grade


def focused_damage_loss(logits, labels):
    return 0.60 * damage_loss(logits, labels) + 0.40 * hierarchical_damage_loss(logits, labels)


def train_earthquake(args, model, device, amp):
    cv2.setNumThreads(1)
    records = json.loads((args.cache / "earthquake.json").read_text(encoding="utf-8"))
    init = args.init or args.run / "building_best.pt"
    previous = load_checkpoint(init, model, device)
    building_epoch = previous.get("building_epoch", previous["epoch"])
    print(f"Initialized from {previous['stage']} epoch {previous['epoch']} ({init})", flush=True)
    data = EarthquakeCrops(args.cache, records["train"], args.seed, args.focus, args.batch)
    objective = focused_damage_loss if args.focus else damage_loss
    new_names = ("fusion.", "damage.", "damage_context.")
    shared, new = [], []
    for name, param in model.named_parameters():
        (new if name.startswith(new_names) else shared).append(param)
    shared_lr, new_lr = (0.00016, 0.0004) if previous["stage"] == "earthquake" else (0.00025, 0.0008)
    optimizer = torch.optim.AdamW([{"params": shared, "lr": shared_lr},
                                  {"params": new, "lr": new_lr}], weight_decay=0.02)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00004)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history, best_score = [], -1.
    start = time.monotonic()
    last_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = torch.zeros(2, device=device)
        steps = 0
        for _ in range(args.steps):
            array = data.batch()
            pre = augment_image(image_tensor(array[..., :3], device), 0.18)
            post = augment_image(image_tensor(array[..., 3:6], device), 0.18)
            labels = tensor_from_batch(array[..., 6], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loc, dmg = model(pre, post)
                loc_loss = footprint_loss(loc, labels != 0)
                dmg_loss = objective(dmg, labels)
                loss = loc_loss + 1.6 * dmg_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            total += torch.stack((loc_loss.detach(), dmg_loss.detach()))
            steps += 1
            if time.time() >= args.deadline:
                break
        schedule.step()
        loc_loss, dmg_loss = (total / steps).tolist()
        if not np.isfinite([loc_loss, dmg_loss]).all():
            raise RuntimeError("Non-finite earthquake loss; retry using --no-amp")
        row = {"epoch": epoch, "steps": steps, "footprint_loss": loc_loss, "damage_loss": dmg_loss}
        validate = epoch in (1, args.epochs) or epoch % 3 == 0 or time.time() >= args.deadline
        if validate:
            thresholds, scores = validation(model, args.cache, records["val"], device, args.batch, amp)
            score = 0.25 * scores["footprint"]["f1"] + 0.75 * scores["damaged_macro_f1"]
            row.update({"validation": scores, "thresholds": thresholds, "selection_score": score})
            if score > best_score:
                best_score, last_improvement = score, epoch
                save_checkpoint(args.run / "damage_best.pt", model, stage="earthquake",
                                classes=CLASSES, tile_size=SIZE, building_epoch=building_epoch,
                                focused=args.focus, initialized_from=str(init.resolve()), **row)
                write_json(args.run / "validation_best.json", row)
        row["elapsed_seconds"] = round(time.monotonic() - start, 1)
        history.append(row)
        write_json(args.run / "damage_history.json", history)
        log = {k: v for k, v in row.items() if k not in ("validation", "thresholds")}
        if validate:
            log.update({"val_footprint_f1": scores["footprint"]["f1"],
                        "val_minor_f1": scores["classes"][CLASSES[2]]["f1"],
                        "val_severe_f1": scores["classes"][CLASSES[3]]["f1"],
                        "thresholds": thresholds})
        print(json.dumps(log), flush=True)
        if time.time() >= args.deadline or (epoch >= 30 and epoch - last_improvement >= 21):
            break
