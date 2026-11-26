"""Supervised damage heads, with thresholds fixed on the development split."""

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn
from torch.nn import functional as F

from dino import WIDTH
from features import STORE, cache_suffix
from inventory import ROOT, write_json

RUN = ROOT / "run"
NAMES = ["no_damage", "minor", "major_or_destroyed"]
QUANTILES = [0.25, .55, .75, .85, .90, .93, .95, .97, .98, .99, .992,
             .994, .996, .998, .999, 1.]


class PairHead(nn.Module):
    def __init__(self, n_features, hidden=96):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(8 * hidden + n_features, hidden),
                                    nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(.40),
                                    nn.Linear(hidden, 3))
        self.register_buffer("center", torch.zeros(n_features))
        self.register_buffer("scale", torch.ones(n_features))

    def joined(self, appearance, change):
        a, b, p, q = self.projection(appearance.float()).unbind(dim=1)
        temporal = [a, b, b - a, (b - a).abs(), p, q, q - p, (q - p).abs()]
        standardized = ((change - self.center) / self.scale).clamp(-8, 8)
        return torch.cat((*temporal, standardized), dim=1)

    def forward(self, appearance, change):
        return self.fusion(self.joined(appearance, change))


def load_data(split, registration="phase", transferred=False):
    suffix = cache_suffix(registration, transferred)
    with np.load(STORE / f"{split}{suffix}.npz", allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def scene_features(values, scenes=None):
    local = np.zeros_like(values)
    groups = (np.arange(len(values)),) if scenes is None else [np.flatnonzero(scenes == k)
                                                               for k in np.unique(scenes)]
    for indices in groups:
        if len(indices) < 8:
            continue
        lo, middle, hi = np.quantile(values[indices], [.25, .5, .75], axis=0)
        local[indices] = ((values[indices] - middle) / np.maximum(hi - lo, .012)).clip(-10, 10)
    return np.concatenate((values, local), axis=1)


def load_head(path, device):
    state = torch.load(path, map_location=device)
    model = PairHead(state["n_features"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model.eval(), state


def matrix(y, predicted):
    return np.bincount(3 * y.astype(np.int64) + predicted, minlength=9).reshape(3, 3)


def report(confusion, unmatched=None):
    truth = confusion.sum(1).copy()
    if unmatched is not None:
        truth += unmatched
    pred = confusion.sum(0)
    hit = confusion.diagonal()
    recall = hit / np.maximum(1, truth)
    precision = hit / np.maximum(1, pred)
    f1 = 2 * hit / np.maximum(1, pred + truth)
    return {"matrix": confusion.tolist(), "unmatched_truth": (
                np.zeros(3, dtype=np.int64) if unmatched is None else unmatched).tolist(),
            "accuracy": float(hit.sum() / max(1, truth.sum())),
            "macro_f1": float(f1.mean()),
            "damaged_macro_f1": float(f1[1:].mean()),
            "balanced_accuracy": float(recall.mean()),
            "classes": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                               "f1": float(f1[i]), "support": int(truth[i]),
                               "predicted": int(pred[i])} for i, name in enumerate(NAMES)}}


def odds(scores):
    logs = np.log(scores.clip(1e-8))
    return logs[:, 1:] - logs[:, :1]


def predict_grade(scores, thresholds):
    margins = odds(scores) - thresholds
    strongest = margins.argmax(1)
    return np.where(margins.max(1) >= 0, strongest + 1, 0)


def calibrate(scores, grades):
    margins = odds(scores)
    grids = [np.unique(np.r_[-16., np.quantile(margins[:, i], QUANTILES), 16.])
             for i in range(2)]
    best_score, choice, values = -1., None, None
    for light in grids[0]:
        for severe in grids[1]:
            predicted = predict_grade(scores, np.array([light, severe]))
            results = report(matrix(grades, predicted))
            score = results["macro_f1"]
            if score > best_score:
                best_score, choice, values = score, [float(light), float(severe)], results
    return choice, values


@torch.no_grad()
def probabilities(model, visual, numeric, device, batch=512):
    model.eval()
    scores = []
    for start in range(0, len(visual), batch):
        a = torch.as_tensor(visual[start:start + batch], device=device)
        x = torch.as_tensor(numeric[start:start + batch], device=device)
        scores.append(model(a, x).softmax(-1).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros((0, 3), np.float32)


def save_model(path, model, **config):
    stage = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "n_features": model.center.numel(), **config}, stage)
    stage.replace(path)


def hard_damage_loss(logits, grade):
    losses = []
    for target in (1, 2):
        other = [k for k in range(3) if k != target]
        margin = logits[:, target] - logits[:, other].logsumexp(1)
        positive = margin[grade == target]
        negative = margin[grade == 0].topk(24).values
        losses.append(F.softplus(negative[None] - positive[:, None]).mean())
    return torch.stack(losses).mean()


def train_head(train, val, device, out, epochs=42, auxiliary=None, external_severe=None,
               hard_negatives=False, auxiliary_strength=.65):
    n_features = train["change"].shape[1]
    rng = np.random.default_rng(641587)
    torch.manual_seed(4081)
    model = PairHead(n_features).to(device)
    model.center.copy_(torch.from_numpy(train["change"].mean(0)))
    model.scale.copy_(torch.from_numpy(train["change"].std(0).clip(.015)))
    groups = [np.flatnonzero(train["grade"] == k) for k in range(3)]
    optim = torch.optim.AdamW(model.parameters(), lr=.00065, weight_decay=.065)
    visual = torch.as_tensor(train["appearance"], device=device)
    numeric = torch.as_tensor(train["change"], device=device)
    grade = torch.as_tensor(train["grade"].astype(np.int64), device=device)
    if auxiliary is not None:
        aux_visual = torch.as_tensor(auxiliary["appearance"], device=device)
        aux_numeric = torch.as_tensor(auxiliary["change"], device=device)
        aux_target = torch.as_tensor(auxiliary["grade"].astype(np.int64), device=device)
        aux_features = model.fusion[:-1]
        shared_classifier = nn.Linear(model.fusion[-1].in_features, 3).to(device)
        optim.add_param_group({"params": shared_classifier.parameters(), "lr": .0007})
    if external_severe is not None:
        severe_visual = torch.as_tensor(external_severe["appearance"], device=device)
        severe_change = torch.as_tensor(external_severe["change"], device=device)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max(1, epochs), eta_min=.00007)
    best = -1.
    history = []
    for epoch in range(1, epochs + 1):
        hard = None
        if hard_negatives and epoch > 1:
            intact = groups[0]
            scores = probabilities(model, train["appearance"][intact], train["change"][intact], device)
            ranking = np.argsort(odds(scores), axis=0)
            hard = intact[np.unique(ranking[-min(850, len(intact)):])]
        model.train()
        running = 0.
        for _ in range(44):
            ids = np.concatenate([rng.choice(g, size, replace=True)
                                  for g, size in zip(groups, (112, 32, 24))])
            if hard is not None:
                ids[:56] = rng.choice(hard, 56, replace=True)
            ids = torch.as_tensor(ids, device=device)
            optim.zero_grad(set_to_none=True)
            logits = model(visual[ids], numeric[ids])
            loss = F.cross_entropy(logits, grade[ids], label_smoothing=.025)
            if hard_negatives:
                loss = loss + .25 * hard_damage_loss(logits, grade[ids])
            if auxiliary is not None:
                aux_ids = torch.as_tensor(rng.integers(0, len(aux_target), 100), device=device)
                aux_logits = shared_classifier(aux_features(model.joined(aux_visual[aux_ids],
                                                                        aux_numeric[aux_ids])))
                loss = loss + auxiliary_strength * F.cross_entropy(aux_logits, aux_target[aux_ids], label_smoothing=.04)
            if external_severe is not None:
                af_ids = torch.as_tensor(rng.integers(0, len(severe_visual), 28), device=device)
                af_scores = model(severe_visual[af_ids], severe_change[af_ids])
                af_target = torch.full((len(af_ids),), 2, device=device, dtype=torch.long)
                loss = loss + .12 * F.cross_entropy(af_scores, af_target, label_smoothing=.08)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optim.step()
            running += float(loss.detach())
        schedule.step()
        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            scores = probabilities(model, val["appearance"], val["change"], device)
            thresholds, results = calibrate(scores, val["grade"])
            if results["macro_f1"] > best:
                best = results["macro_f1"]
                save_model(out / "head.pt", model, epoch=epoch, thresholds=thresholds,
                           validation=results, auxiliary=auxiliary is not None,
                           additional_severe=external_severe is not None, hard_negatives=hard_negatives)
                np.save(out / "val_head_scores.npy", scores)
            history.append({"epoch": epoch, "loss": running / 44, "validation": results})
            write_json(out / "training.json", history)
            print(json.dumps({"epoch": epoch, "loss": round(running / 44, 3),
                              "val_macro_f1": results["macro_f1"],
                              "damaged_macro_f1": results["damaged_macro_f1"]}), flush=True)
    return np.load(out / "val_head_scores.npy")


def train_texture(train, val, out, kernel=False):
    values = train["grade"]
    if kernel:
        rng = np.random.default_rng(6388)
        intact = np.flatnonzero(values == 0)
        include = np.r_[rng.choice(intact, min(2700, len(intact)), replace=False),
                        np.flatnonzero(values != 0)]
        method = make_pipeline(StandardScaler(), SVC(C=2.2, class_weight="balanced", probability=True,
                                                      random_state=187, cache_size=360))
        method.fit(train["change"][include], values[include])
    else:
        proportions = np.bincount(values, minlength=3).clip(1) / len(values)
        weights = (1 / proportions[values]) ** .66
        method = HistGradientBoostingClassifier(max_iter=145, max_depth=4, max_leaf_nodes=15,
                                                 min_samples_leaf=19, learning_rate=.075,
                                                 l2_regularization=2., random_state=108)
        method.fit(train["change"], values, sample_weight=weights)
    scores = method.predict_proba(val["change"])
    joblib.dump(method, out / "texture.joblib")
    np.save(out / "val_texture_scores.npy", scores)
    return scores


def choose(val, head_p, texture_p, out, use_context=False, registration="phase", transferred=False):
    rows = []
    for blend in (0., .20, .40, .65, 1.):
        scores = (1 - blend) * head_p + blend * texture_p
        thresholds, results = calibrate(scores, val["grade"])
        rows.append({"texture_weight": blend, "thresholds": thresholds, "validation": results})
    best = max(rows, key=lambda row: row["validation"]["macro_f1"])
    write_json(out / "selection.json", {"selected": best, "candidates": rows,
                                        "scene_context": use_context, "registration": registration,
                                        "transferred_encoder": transferred})
    print(json.dumps(best, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=RUN / "mexico")
    cli.add_argument("--epochs", type=int, default=42)
    cli.add_argument("--aux", type=Path)
    cli.add_argument("--aux-strength", type=float, default=.65)
    cli.add_argument("--extra-severe", type=Path)
    cli.add_argument("--scene-context", action="store_true")
    cli.add_argument("--device", choices=("auto", "cpu"), default="auto")
    cli.add_argument("--registration", choices=("phase", "keypoint"), default="phase")
    cli.add_argument("--transfer", action="store_true")
    cli.add_argument("--rbf", action="store_true")
    cli.add_argument("--hard-negatives", action="store_true")
    cli.add_argument("--augment-training", action="store_true", help="Append matching jittered training crops")
    args = cli.parse_args()
    if args.epochs < 1:
        cli.error("epochs must be positive")
    if not 0. < args.aux_strength <= 1.:
        cli.error("aux-strength must be above 0 and at most 1")
    if args.extra_severe and args.transfer:
        cli.error("the external patches use the original encoder")
    torch.set_num_threads(3)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    train, val = [load_data(split, args.registration, args.transfer) for split in ("train", "val")]
    if args.augment_training:
        suffix = cache_suffix(args.registration, args.transfer)
        with np.load(STORE / f"train{suffix}_jittered.npz", allow_pickle=False) as arrays:
            train = {key: np.concatenate((value, arrays[key])) for key, value in train.items()}
    if args.aux:
        with np.load(args.aux, allow_pickle=False) as arrays:
            auxiliary = {k: arrays[k] for k in ("appearance", "change", "grade", "chip")}
    else:
        auxiliary = None
    if args.extra_severe:
        with np.load(args.extra_severe, allow_pickle=False) as arrays:
            if "grade" in arrays.files and not np.all(arrays["grade"] == 2):
                cli.error("extra-severe must contain only severe or destroyed buildings")
            external_severe = {k: arrays[k] for k in ("appearance", "change", "chip")}
    else:
        external_severe = None
    if args.scene_context:
        for data in (train, val, auxiliary, external_severe):
            if data is not None:
                data["change"] = scene_features(data["change"], data["chip"])
    head = train_head(train, val, device, args.out, args.epochs, auxiliary, external_severe,
                      args.hard_negatives, args.aux_strength)
    texture = train_texture(train, val, args.out, args.rbf)
    choose(val, head, texture, args.out, args.scene_context, args.registration, args.transfer)
    print(f"Classifier time {time.monotonic() - start:.1f}s", flush=True)
