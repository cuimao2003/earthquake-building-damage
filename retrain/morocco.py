"""Prepare paired optical crops on buildings graded by Copernicus EMS in Morocco."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
from functools import lru_cache
import gzip
import json
import math
import os

import cv2
import numpy as np
from osgeo import gdal
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import transform

from inventory import ROOT, write_json


DATA = ROOT / "data/external/morocco"
FEATURES = ROOT / "data/features"
EVENT = date(2023, 9, 8)
GRADES = {"Damaged": 1, "Destroyed": 2}
SIZE = 112
GRID = .0025


def survey():
    grades, coordinates = Counter(), defaultdict(list)
    for p in sorted(DATA.glob("aoi*-builtUpP.geojson")):
        for feature in json.loads(p.read_bytes())["features"]:
            geom, props = feature["geometry"], feature["properties"]
            if geom is None or geom["type"] != "Point":
                continue
            label = props["damage_gra"]
            grades[label] += 1
            if label in GRADES:
                x, y = geom["coordinates"][:2]
                coordinates[(round(x, 6), round(y, 6))].append((p.name[:5], label))
    points = []
    distinct = 0
    for (x, y), annotations in coordinates.items():
        if len({g for _, g in annotations}) != 1:
            distinct += 1
            continue
        aoi, label = min(annotations)
        points.append({"point": [x, y], "aoi": aoi, "grade": GRADES[label], "cems_label": label})
    return points, {"all_grades": grades, "conflicting_coordinates": distinct, "eligible_points": len(points)}


def cell(x, y):
    return math.floor(x / GRID), math.floor(y / GRID)


@lru_cache(maxsize=8)
def projection(epsg):
    return Transformer.from_crs(4326, epsg, always_xy=True)


def match_footprints(points):
    source = DATA / "footprints.csv.gz"
    last = DATA / "building_matches.json"
    if last.is_file():
        return json.loads(last.read_bytes())
    groups = defaultdict(list)
    for i, row in enumerate(points):
        longitude, latitude = row["point"]
        gx, gy = cell(longitude, latitude)
        for x in range(gx - 1, gx + 2):
            for y in range(gy - 1, gy + 2):
                groups[x, y].append(i)
    matches = defaultdict(list)
    with gzip.open(source, "rt", encoding="utf-8", newline="") as archive:
        for n, row in enumerate(csv.DictReader(archive), 1):
            if n % 1500000 == 0:
                print(f"Scanned {n:,} Open Buildings footprints", flush=True)
            x, y = float(row["longitude"]), float(row["latitude"])
            candidates = groups.get(cell(x, y), ())
            date_code = row["last_detection_date"]
            if not candidates or float(row["confidence"]) < .65 or not date_code or date_code >= "2023-09":
                continue
            roof = wkt.loads(row["geometry"])
            if roof.is_empty or not roof.is_valid:
                continue
            for i in candidates:
                point = Point(*points[i]["point"])
                if roof.covers(point):
                    matches[i].append(row["geometry"])
    matched = []
    occupied = defaultdict(list)
    for i, candidates in matches.items():
        if len(set(candidates)) == 1:
            occupied[candidates[0]].append(points[i])
    for geometry, sources in occupied.items():
        if len({p["grade"] for p in sources}) == 1:
            matched.append({**min(sources, key=lambda p: (p["aoi"], p["point"])), "geometry": geometry})
    write_json(last, matched)
    print(f"CEMS points matched to {len(matched)} distinct model-derived roof footprints", flush=True)
    return matched


class Catalog:
    def __init__(self):
        self.rows = defaultdict(list)
        with (DATA / "maxar_catalog.tsv").open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                day = date.fromisoformat(row["datetime"][:10])
                clouds = float(row["tile:clouds_percent"])
                if day < date(2019, 1, 1) or day > date(2023, 10, 8) or clouds > 18:
                    continue
                epsg = int(row["proj:epsg"])
                bounds = tuple(map(float, row["proj:bbox"].split(",")))
                sorted_date = abs((day - EVENT).days)
                self.rows[epsg, day > EVENT].append((bounds, row["visual"], day.isoformat(),
                                                     sorted_date, clouds, float(row["view:off_nadir"])))
        self.bounds = {key: np.array([r[0] for r in rows]) for key, rows in self.rows.items()}

    def nearby(self, bounds, epsg, post):
        values = self.bounds.get((epsg, post))
        if values is None:
            return []
        x0, y0, x1, y1 = bounds
        exact = ((values[:, 0] <= x0) & (values[:, 1] <= y0) &
                 (values[:, 2] >= x1) & (values[:, 3] >= y1))
        candidates = [self.rows[epsg, post][i] for i in np.flatnonzero(exact)]
        return sorted(candidates, key=lambda r: (r[3] + 2 * r[4] + r[5]))[:4]


@lru_cache(maxsize=32)
def raster(url):
    return gdal.Open("/vsicurl/" + url)


def fetch_patch(candidates, bounds):
    x0, y0, x1, y1 = bounds
    for _, url, day, *_ in candidates:
        try:
            ds = raster(url)
            inverse = gdal.InvGeoTransform(ds.GetGeoTransform())
            left, top = gdal.ApplyGeoTransform(inverse, x0, y1)
            right, bottom = gdal.ApplyGeoTransform(inverse, x1, y0)
            x, y = int(round(left)), int(round(top))
            width, height = int(round(right)) - x, int(round(bottom)) - y
            if not (x >= 0 and y >= 0 and x + width <= ds.RasterXSize and y + height <= ds.RasterYSize):
                continue
            bands = [1, 2, 3]
            if ds.RasterCount >= 4:
                bands.append(4)
            values = ds.ReadAsArray(x, y, width, height, buf_xsize=SIZE, buf_ysize=SIZE, band_list=bands,
                                    resample_alg=gdal.GRIORA_Bilinear)
            if values.shape[0] > 3 and (values[3] < 250).mean() > .02:
                continue
            image = np.ascontiguousarray(np.moveaxis(values[:3], 0, -1))
            if (image.max(2) == 0).mean() > .02 or image.std() < 4:
                continue
            return image, url, day
        except RuntimeError:
            continue
    return None


def make_patch(row, catalog):
    x, y = row["point"]
    epsg = 32600 + math.floor((x + 180) / 6) + 1
    proj = projection(epsg)
    if "geometry" in row:
        roof = transform(proj.transform, wkt.loads(row["geometry"]))
        x0, y0, x1, y1 = roof.bounds
        px, py = (x0 + x1) / 2, (y0 + y1) / 2
        width, height = x1 - x0, y1 - y0
    else:
        px, py = proj.transform(x, y)
        width = height = 10.
    extent = float(np.clip(max(width, height) * 1.8, 17., 78.))
    half = extent / 2
    bounds = (px - half, py - half, px + half, py + half)
    before, after = [catalog.nearby(bounds, epsg, post) for post in (False, True)]
    if not before or not after:
        return None, "no_catalog_pair"
    pre, post = [fetch_patch(v, bounds) for v in (before, after)]
    if pre is None or post is None:
        return None, "unreadable_or_invalid_crop"
    limit = np.clip(np.array([width, height]) * .5 * SIZE / extent, 1.5, SIZE * .48)
    yp, xp = np.ogrid[:SIZE, :SIZE]
    mask = ((xp - SIZE / 2) / limit[0]) ** 2 + ((yp - SIZE / 2) / limit[1]) ** 2 <= 1
    return (np.stack((pre[0], post[0])), mask, {"pre": pre[1:], "post": post[1:]}), None


def make_crops(points, name, limit):
    from features import align, signal

    clips = DATA / f"{name}_crops.npz"
    if clips.is_file():
        print(f"Using {clips.name}", flush=True)
        return
    catalog = Catalog()
    images, regions, change, accepted = [], [], [], []
    skipped = Counter()
    points = sorted(points, key=lambda r: (r["aoi"], r["point"]))
    if limit:
        points = [points[i] for i in np.linspace(0, len(points) - 1, min(limit, len(points)), dtype=int)]
    for i, row in enumerate(points, 1):
        pair, reason = make_patch(row, catalog)
        if pair is None:
            skipped[reason] += 1
        else:
            rgb, roi, source = pair
            rgb[1] = align(*rgb, "phase")[0]
            images.append(rgb)
            regions.append(roi)
            change.append(signal(rgb, roi))
            accepted.append({**{k: v for k, v in row.items() if k != "geometry"}, **source})
        if i % 40 == 0 or i == len(points):
            print(f"Morocco optical crops {i}/{len(points)}, {len(accepted)} valid", flush=True)
    report = {"attempted": len(points), "accepted": len(accepted), "skipped": skipped,
              "source_labels": "https://mapping.emergency.copernicus.eu/activations/EMSR695/",
              "imagery": "Maxar Open Data ARD", "counts_by_grade": Counter(p["cems_label"] for p in accepted),
              "counts_by_aoi": Counter(p["aoi"] for p in accepted), "roi": name}
    write_json(DATA / f"{name}_crop_audit.json", report)
    write_json(DATA / f"{name}_crop_sources.json", accepted)
    print(json.dumps(report, indent=2), flush=True)
    if not accepted:
        raise ValueError("No valid pairs were available")
    np.savez(clips, rgb=np.stack(images), roi=np.stack(regions), texture=np.stack(change),
             grade=np.array([p["grade"] for p in accepted], np.uint8), chip=np.array([p["aoi"] for p in accepted]))


def embed_crops(name):
    import torch
    from dino import load_dino
    from features import WEIGHTS, embed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    model = load_dino(WEIGHTS, device)
    z = np.load(DATA / f"{name}_crops.npz", allow_pickle=False)
    rgb, roi, texture = (z[k] for k in ("rgb", "roi", "texture"))
    visual, numeric = [], []
    for start in range(0, len(rgb), 72):
        end = start + 72
        a, b = embed(model, rgb[start:end], roi[start:end], device)
        visual.append(a.astype(np.float16))
        numeric.append(np.column_stack((texture[start:end], b)))
    arrays = {"appearance": np.concatenate(visual), "change": np.concatenate(numeric),
              "grade": z["grade"], "chip": z["chip"]}
    np.savez(FEATURES / f"morocco_{name}.npz", **arrays)
    destroyed = arrays["grade"] == GRADES["Destroyed"]
    if destroyed.any():
        np.savez(FEATURES / f"morocco_{name}_destroyed.npz",
                 **{key: value[destroyed] for key, value in arrays.items()})


def diagnose_grades(name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    with np.load(FEATURES / f"morocco_{name}.npz", allow_pickle=False) as z:
        raw, change, target, groups = (z[k] for k in ("appearance", "change", "grade", "chip"))
    a, b, p, q = raw.astype(np.float32).transpose(1, 0, 2)
    visual = np.concatenate((b, q, b - a, q - p), axis=1)
    predicted = np.zeros(len(target), dtype=np.int8)
    for train, test in LeaveOneGroupOut().split(visual, target, groups):
        transformer = make_pipeline(StandardScaler(), SVC(C=1.7, class_weight="balanced", probability=True,
                                                          random_state=699))
        numerical = HistGradientBoostingClassifier(max_iter=75, max_leaf_nodes=10, min_samples_leaf=14,
                                                    l2_regularization=2., random_state=699)
        transformer.fit(visual[train], target[train])
        numerical.fit(change[train], target[train])
        posterior = .70 * transformer.predict_proba(visual[test]) + .30 * numerical.predict_proba(change[test])
        predicted[test] = transformer.classes_[posterior.argmax(1)]
    report = {"protocol": "Leave one Copernicus area out per fold, supplied roof footprints",
              "labels": ["Damaged", "Destroyed"], "support": np.bincount(target, minlength=3)[1:].tolist(),
              "zones": sorted(set(groups)), "accuracy": float((predicted == target).mean()),
              "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
              "macro_f1": float(f1_score(target, predicted, average="macro")),
              "confusion": confusion_matrix(target, predicted, labels=[1, 2]).tolist()}
    write_json(DATA / f"{name}_severity_diagnostic.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("inspect", "crops", "embed", "diagnose"))
    cli.add_argument("--roi", choices=("footprints", "centered"), default="footprints")
    cli.add_argument("--limit", type=int)
    args = cli.parse_args()
    cv2.setNumThreads(1)
    gdal.UseExceptions()
    gdal.SetCacheMax(192 * 1024 * 1024)
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    gdal.SetConfigOption("GDAL_HTTP_CONNECTTIMEOUT", "8")
    gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", "24")
    label = f"{args.roi}_pilot{args.limit}" if args.limit else args.roi
    if args.action == "embed":
        embed_crops(label)
    elif args.action == "diagnose":
        diagnose_grades(label)
    else:
        rows, details = survey()
        print(json.dumps(details, indent=2, ensure_ascii=False), flush=True)
        write_json(DATA / "survey_audit.json", details)
        if args.action == "crops":
            selected = match_footprints(rows) if args.roi == "footprints" else rows
            make_crops(selected, label, args.limit)
