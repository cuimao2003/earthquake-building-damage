"""Learn the expected two-date appearance of undamaged roofs."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
from torch import nn
from torch.nn import functional as F

from classifier import calibrate, load_data, scene_features
from dino import WIDTH
from inventory import ROOT, write_json


class IntactChange(nn.Module):
    def __init__(self):
        super().__init__()
        width = 2 * WIDTH
        self.layers = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, 256),
                                    nn.GELU(), nn.Dropout(.20), nn.Linear(256, width))

    def forward(self, pre, style):
        return pre + style + .45 * self.layers(torch.cat((pre, style), dim=1))


def arrange(appearance, groups):
    before = appearance[:, (0, 2)].reshape(len(appearance), -1).astype(np.float32) * np.sqrt(WIDTH)
    after = appearance[:, (1, 3)].reshape(len(appearance), -1).astype(np.float32) * np.sqrt(WIDTH)
    diff = after - before
    styles = np.zeros_like(before)
    members = [np.arange(len(before))] if groups is None else [np.flatnonzero(groups == key)
                                                               for key in np.unique(groups)]
    for group in members:
        styles[group] = np.median(diff[group], axis=0)
    return before, after, styles


@torch.no_grad()
def features(model, appearance, signals, groups=None, device="cpu"):
    pre, post, styles = arrange(appearance, groups)
    model.eval()
    expected = np.concatenate([model(torch.from_numpy(pre[i:i + 768]).to(device),
                                     torch.from_numpy(styles[i:i + 768]).to(device)).cpu().numpy()
                               for i in range(0, len(pre), 768)])
    residuals = [np.abs(post - expected), np.abs(post - pre - styles), np.abs(post - pre)]
    stats = []
    for residual in residuals:
        for channel in np.split(residual, 2, axis=1):
            stats.append(np.column_stack((channel.mean(1), channel.std(1),
                                          np.quantile(channel, [.75, .90, .98], axis=1).T)))
    return np.concatenate((scene_features(signals, groups), scene_features(np.concatenate(stats, 1), groups)), 1)


def train(args):
    torch.manual_seed(426113)
    rng = np.random.default_rng(4729)
    train, val = [load_data(split, args.registration, args.transfer) for split in ("train", "val")]
    model = IntactChange()
    opt = torch.optim.AdamW(model.parameters(), lr=.00045, weight_decay=.035)
    a, b, s = [torch.from_numpy(p) for p in arrange(train["appearance"], train["chip"])]
    intact = np.flatnonzero(train["grade"] == 0)
    for epoch in range(24):
        model.train()
        running = 0.
        for _ in range(58):
            indices = rng.choice(intact, min(len(intact), 128), replace=False)
            opt.zero_grad(set_to_none=True)
            err = F.smooth_l1_loss(model(a[indices], s[indices]), b[indices], beta=.35)
            err.backward()
            opt.step()
            running += float(err.detach())
        if (epoch + 1) % 6 == 0:
            print(f"Intact predictor {epoch + 1}/24, loss {running / 58:.4f}", flush=True)
    train_x = features(model, train["appearance"], train["change"], train["chip"])
    val_x = features(model, val["appearance"], val["change"], val["chip"])
    y = train["grade"]
    weights = np.power(len(y) / np.bincount(y, minlength=3).clip(1)[y], .63)
    candidates = []
    best = -1.
    args.out.mkdir(parents=True, exist_ok=True)
    for iterations, depth in ((100, 3), (180, 4), (210, 6)):
        clf = HistGradientBoostingClassifier(max_depth=depth, max_iter=iterations, learning_rate=.06,
                                              min_samples_leaf=18, l2_regularization=1.7,
                                              random_state=42291)
        clf.fit(train_x, y, sample_weight=weights)
        scores = clf.predict_proba(val_x)
        thresholds, metrics = calibrate(scores, val["grade"])
        row = {"thresholds": thresholds, "validation": metrics, "iterations": iterations, "depth": depth}
        candidates.append(row)
        print(f"Intact prediction features depth={depth}: {metrics['macro_f1']:.4f}", flush=True)
        if metrics["macro_f1"] > best:
            selected, best = row, metrics["macro_f1"]
            joblib.dump(clf, args.out / "hazard.joblib")
            np.save(args.out / "val_anomaly_scores.npy", scores)
    torch.save(model.state_dict(), args.out / "intact.pt")
    write_json(args.out / "selection.json", {"selected": selected, "candidates": candidates,
                                             "registration": args.registration,
                                             "transferred_encoder": args.transfer})
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=ROOT / "run/mexico_intact")
    cli.add_argument("--registration", default="keypoint", choices=("phase", "keypoint"))
    cli.add_argument("--transfer", action="store_true")
    args = cli.parse_args()
    torch.set_num_threads(3)
    train(args)
