"""Collect all cross-validated metrics used for the 2026-11-27 mixed-data report."""

import json
from pathlib import Path

RUN = Path(r"D:\土坑论文\earthquake_transformer\retrain\run")
OUT = RUN / "final_metrics.json"


def read(relative):
    path = RUN / relative
    return json.loads(path.read_bytes()) if path.is_file() else None


def val_metrics(relative):
    data = read(relative)
    if not data:
        return None
    validation = data["validation"]
    return {
        "macro_f1": round(validation["macro_f1"], 4),
        "damaged_macro_f1": round(validation["damaged_macro_f1"], 4),
        "minor_f1": round(validation["classes"]["minor"]["f1"], 4),
        "minor_tp": validation["matrix"][1][1],
        "minor_predicted": validation["classes"]["minor"]["predicted"],
        "major_f1": round(validation["classes"]["major_or_destroyed"]["f1"], 4),
        "major_tp": validation["matrix"][2][2],
        "major_predicted": validation["classes"]["major_or_destroyed"]["predicted"],
        "box_f1": round(data.get("box_f1", 0), 4) if data.get("box_f1") else None,
    }


def fusion_metrics(relative):
    data = read(relative)
    if not data:
        return None
    selected = data["selected"]
    return {
        "weight": selected["weight"],
        "macro_f1": round(selected["validation"]["macro_f1"], 4),
        "damaged_macro_f1": round(selected["validation"]["damaged_macro_f1"], 4),
    }


def end_to_end(relative):
    data = read(relative)
    if not data:
        return None
    classification = data["classification"]
    return {
        "macro_f1": round(classification["macro_f1"], 4),
        "damaged_macro_f1": round(classification["damaged_macro_f1"], 4),
        "accuracy": round(classification["accuracy"], 4),
        "box_f1": round(data["box_f1"], 4),
        "major_f1": round(classification["classes"]["major_or_destroyed"]["f1"], 4),
        "major_tp": classification["matrix"][2][2],
        "major_predicted": classification["classes"]["major_or_destroyed"]["predicted"],
        "given_box_accuracy": round(data["given_human_boxes_diagnostic"]["accuracy"], 4)
        if "given_human_boxes_diagnostic" in data else None,
        "given_box_major_f1": round(data["given_human_boxes_diagnostic"]["classes"]["major_or_destroyed"]["f1"], 4)
        if "given_human_boxes_diagnostic" in data else None,
    }


def kate_eval(relative):
    data = read(relative)
    return None if not data else {
        key: round(data[key], 4) for key in ("pixel_damage_precision", "pixel_damage_recall", "pixel_damage_f1", "iou")
    }


def bootstrap(relative):
    data = read(relative)
    if not data:
        return None
    return {"point": {key: round(value, 4) for key, value in data["point"].items()},
            "macro_f1": {key: round(value, 4) for key, value in data["macro_f1"].items()},
            "damaged_macro_f1": {key: round(value, 4) for key, value in data["damaged_macro_f1"].items()}}


summary = {
    "mexico_validation": {
        "baseline_old_model": "0.2296 (see mexico_large_selection/automatic_thresholds.json)",
        "mix_palu_pooled": val_metrics("mix_palu_pooled/validation.json"),
        "mix_palu_attention": val_metrics("mix_palu_attention/validation.json"),
        "mix_palu_rank": val_metrics("mix_palu_rank/validation.json"),
        "mix_palu_kate": val_metrics("mix_palu_kate/validation.json"),
        "proposal_attention_floor24": val_metrics("proposal_attention_floor24/validation.json"),
    },
    "mexico_fusion_validation": {
        "mix_fusion": fusion_metrics("mix_fusion/fusion.json"),
        "floor24_fusion": fusion_metrics("floor24_fusion/fusion.json"),
    },
    "mexico_history_holdout": {
        "old_model": 0.1557,
        "mix_palu_pooled": end_to_end("mix_palu_regression/end_to_end.json"),
        "mix_fusion": end_to_end("mix_fusion_regression/end_to_end.json"),
        "floor24_fusion": end_to_end("floor24_fusion_regression/end_to_end.json"),
    },
    "palu_cross_event": {
        "old_model": end_to_end("cross_event_palu_old/end_to_end.json"),
        "floor24_fusion": end_to_end("cross_event_palu_floor24_fusion/end_to_end.json"),
        "mix_palu_pooled": end_to_end("cross_event_palu_mix_pooled/end_to_end.json"),
        "mix_palu_attention": end_to_end("cross_event_palu_mix_attention/end_to_end.json"),
        "mix_palu_rank": end_to_end("cross_event_palu_mix_rank/end_to_end.json"),
        "mix_palu_kate": end_to_end("cross_event_palu_mix_kate/end_to_end.json"),
        "mix_fusion": end_to_end("cross_event_palu_mix_fusion/end_to_end.json"),
    },
    "kate_cd_turkey2023": {
        "old_model": kate_eval("kate_eval_old/kate_eval.json"),
        "mix_palu_pooled": kate_eval("kate_eval_mix_pooled/kate_eval.json"),
        "mix_palu_kate": kate_eval("kate_eval_mix_kate/kate_eval.json"),
    },
    "bootstrap_validation": {
        "mix_palu_pooled": bootstrap("bootstrap_mix_palu_pooled.json"),
        "mix_palu_kate": bootstrap("bootstrap_mix_palu_kate.json"),
    },
}
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
