"""Compare an intermediate-grade margin classifier using the learned ViT features."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch

from classifier import calibrate, matrix, predict_grade, report
from features import descriptors
from finetune_turkey import Patches, load, test_examples
from inventory import ROOT, write_json


@torch.no_grad()
def encode(path, split, device):
    encoder, head, state = load(path, device)
    size = state.get("patch_size", 112)
    evidence = state.get("local_evidence", False)
    dataset = Patches(split, size)
    vectors, signals, raw = [], [], []
    for i in range(0, len(dataset.grade), 48):
        rgb, roi, statistics, _ = dataset.batch(slice(i, i + 48), device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            a, d = descriptors(encoder, rgb, roi, evidence)
            change = torch.cat((statistics, d), dim=1)
            scores = head(a, change).float().softmax(-1)
        pre, post, region_pre, region_post = a.unbind(1)
        vectors.append(torch.cat((pre, post, post - pre, region_pre, region_post,
                                  region_post - region_pre), dim=1).cpu().numpy())
        signals.append(change.cpu().numpy())
        raw.append(scores.cpu().numpy())
    return dataset, np.concatenate(vectors), np.concatenate(signals), np.concatenate(raw)


def transformed(visual, numeric, pca, scaler):
    compressed = pca.transform(visual) / np.sqrt(pca.n_components_)
    texture = scaler.transform(numeric).clip(-7, 7) / np.sqrt(numeric.shape[1])
    return np.concatenate((compressed, texture), axis=1)


def fit(args, device):
    train, ap, raw, _ = encode(args.checkpoint, "train", device)
    pca = PCA(n_components=72, whiten=True, svd_solver="randomized", random_state=798).fit(ap)
    scale = StandardScaler().fit(raw)
    x = transformed(ap, raw, pca, scale)
    valid, features, numeric, base = encode(args.checkpoint, "val", device)
    held = transformed(features, numeric, pca, scale)
    top = None
    candidates = []
    for c in (.7, 2.4, 6.):
        svm = SVC(C=c, class_weight="balanced", probability=True, gamma="scale", cache_size=420,
                  random_state=56)
        svm.fit(x, train.grade)
        extra = svm.predict_proba(held)
        for weight in (.20, .45, .70, 1.):
            cuts, performance = calibrate((1 - weight) * base + weight * extra, valid.grade)
            row = {"c": c, "svm_weight": weight, "thresholds": cuts, "validation": performance}
            candidates.append(row)
            if top is None or performance["macro_f1"] > top[0]["validation"]["macro_f1"]:
                top = row, svm
        print(f"Pairwise RBF C={c}: best validation macro F1 {top[0]['validation']['macro_f1']:.4f}", flush=True)
    joblib.dump({"pca": pca, "scale": scale, "svm": top[1]}, args.out / "expert.joblib")
    write_json(args.out / "selection.json", {"selected": top[0], "candidates": candidates,
                                             "checkpoint": str(args.checkpoint.resolve())})


def test(args, device):
    configuration = json.loads((args.out / "selection.json").read_bytes())
    selected = configuration["selected"]
    source = Path(configuration["checkpoint"])
    model = joblib.load(args.out / "expert.joblib")
    data, features, numeric, base = encode(source, "test", device)
    inputs = transformed(features, numeric, model["pca"], model["scale"])
    scores = (1 - selected["svm_weight"]) * base + selected["svm_weight"] * model["svm"].predict_proba(inputs)
    predicted = predict_grade(scores, np.array(selected["thresholds"]))
    metrics = report(matrix(data.grade, predicted))
    metrics["grade_names"] = ["A", "B", "C"]
    metrics["test_scenes"] = sorted(set(data.chip.tolist()))
    metrics["settings"] = selected
    write_json(args.out / "test.json", metrics)
    test_examples(data, predicted, args.out / "test_examples.png")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "run/turkey_vit168/turkey_vit.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "run/turkey_expert")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (test if args.test else fit)(args, device)
