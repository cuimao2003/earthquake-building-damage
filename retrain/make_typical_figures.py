"""Draw the reference-style four-panel figures for the three best Turkey test scenes."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from classifier import predict_grade
from detector import match
from inventory import ROOT
from turkey import DATA, load_regions
from turkey_automatic import Assessment, OUT, detection_image, paired

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEX = ["#52b3e8", "#ffc231", "#ff4855"]
CLASS_TEXT = ("A 屋顶无明显损伤", "B 部分损伤", "C 整体倒塌")
SCENES = (53, 59, 4)
CAPTIONS = {
    53: "图1 土耳其震区 53 场景的人工标注与自动分级",
    59: "图2 土耳其震区 59 场景的人工标注与自动分级",
    4: "图3 土耳其震区 04 场景的人工标注与自动分级",
}
DISPLAY_WIDTH = 1500
OUT_DIR = ROOT / "run/typical_figures"
CAPTION_FONT = Path("C:/Windows/Fonts/simhei.ttf")
FONT_FAMILY = ["Microsoft YaHei", "SimHei"]

plt.rcParams["font.sans-serif"] = FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False


def display_image(path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = DISPLAY_WIDTH / width
    resized = image.resize((DISPLAY_WIDTH, max(1, round(height * scale))), Image.LANCZOS)
    return np.asarray(resized), scale


def draw_boxes(axis, boxes, colors, linewidth):
    for box, color in zip(boxes, colors):
        left, top, right, bottom = box
        axis.add_patch(plt.Rectangle((left, top), right - left, bottom - top,
                                     fill=False, edgecolor=color, linewidth=linewidth))


def make_figure(number, annotated, predicted, grades, output):
    pre_path = DATA / "pre" / f"{number:02d}.jpg"
    post_path = DATA / "post" / f"{number:02d}.jpg"
    pre, scale = display_image(pre_path)
    post, _ = display_image(post_path)
    truth_boxes = np.asarray([box for box, _ in annotated], np.float32) * scale
    truth_colors = [HEX[grade] for _, grade in annotated]
    pred_boxes = predicted * scale
    pred_colors = [HEX[grade] for grade in grades]

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), facecolor="#141a2a")
    figure.subplots_adjust(left=0.015, right=0.985, top=0.93, bottom=0.105, wspace=0.02, hspace=0.10)
    panels = (
        (axes[0, 0], pre, "(a) 震前影像", None, None),
        (axes[0, 1], post, "(b) 震后影像", None, None),
        (axes[1, 0], pre, "(c) 人工标注", truth_boxes, truth_colors),
        (axes[1, 1], pre, "(d) 自动定位与分级", pred_boxes, pred_colors),
    )
    linewidth = max(0.9, DISPLAY_WIDTH / 1450)
    for axis, image, title, boxes, colors in panels:
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title, loc="left", color="#d7dae3", fontsize=15, pad=7, fontfamily=FONT_FAMILY)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_facecolor("#141a2a")
        if boxes is not None and len(boxes):
            draw_boxes(axis, boxes, colors, linewidth)
    handles = [Patch(facecolor=HEX[i], edgecolor="none", label=CLASS_TEXT[i]) for i in range(3)]
    legend = figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                           fontsize=14, labelcolor="#d7dae3", handlelength=1.05,
                           handleheight=1.05, columnspacing=2.4, bbox_to_anchor=(0.5, 0.012))
    for text in legend.get_texts():
        text.set_fontfamily(FONT_FAMILY)
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def add_caption(figure_path, caption, output_path):
    image = Image.open(figure_path).convert("RGB")
    width, height = image.size
    strip = 92
    canvas = Image.new("RGB", (width, height + strip), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(CAPTION_FONT), 40)
    text_width = draw.textlength(caption, font=font)
    draw.text(((width - text_width) / 2, height + 16), caption, fill=(20, 20, 20), font=font)
    canvas.save(output_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = Assessment(OUT, DEVICE)
    config = json.loads((OUT / "automatic_thresholds.json").read_bytes())
    thresholds = np.asarray(config["thresholds"], dtype=np.float32)
    summary = []
    for number in SCENES:
        before, after = paired(number)
        boxes, scores = model.automatic(before, after, *detection_image(before))
        grades = predict_grade(scores, thresholds)
        annotated = load_regions(number)
        truth = np.asarray([box for box, _ in annotated], np.float32).reshape(-1, 4)
        labels = np.asarray([grade for _, grade in annotated], np.int8)
        matched, assigned = match(boxes, truth)
        correct = int((grades[matched] == labels[assigned]).sum())
        accuracy = correct / max(1, len(labels))
        plain = OUT_DIR / f"scene{number:02d}_typical.png"
        make_figure(number, annotated, boxes, grades, plain)
        captioned = OUT_DIR / f"scene{number:02d}_typical_caption.png"
        add_caption(plain, CAPTIONS[number], captioned)
        with (OUT_DIR / f"scene{number:02d}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["left", "top", "right", "bottom", "grade", "score_A", "score_B", "score_C"])
            for box, grade, values in zip(boxes, grades, scores):
                writer.writerow([*[round(float(v), 2) for v in box], int(grade), *[round(float(v), 4) for v in values]])
        summary.append({"scene": number, "correct": correct, "annotated": int(len(labels)),
                        "accuracy": round(accuracy, 4), "predicted": int(len(boxes)),
                        "matched": int(len(matched)), "figure": str(captioned)})
        print(f"scene {number:02d}: {correct}/{len(labels)} = {accuracy:.4f} -> {captioned.name}", flush=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
