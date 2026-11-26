"""Audit public same-event building survey points against Mexico training roofs."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

from shapely import wkt
from shapely.geometry import Point
from shapely.strtree import STRtree

from inventory import ROOT, write_json


SERVICES = "https://rmgir.proyectomesoamerica.org/server/rest/services/Trans_Sept_2018"
BASE = SERVICES + "/PlataformaCDMX/MapServer/0"
OUT = ROOT / "data/external/cdmx_census"
RISK_NAMES = {
    "RIESGO BAJO": "low_risk",
    "RIESGO MEDIO": "medium_risk",
    "RIESGO ALTO": "high_risk",
    "ALTO RIESGO DE COLAPSO": "high_collapse_risk",
    "COLAPSO PARCIAL": "partial_collapse",
    "COLAPSO TOTAL": "total_collapse",
}


def fetch(layer=BASE, local=OUT / "points.json", oid="OBJECTID", field="TIPO_DE_R_IESGO"):
    if local.is_file():
        return json.loads(local.read_bytes())
    url = layer + "/query"
    points = []
    def query(**fields):
        target = f"{url}?{urlencode({'where': '1=1', 'f': 'json', **fields})}"
        response = subprocess.check_output(["curl.exe", "-fsS", "--retry", "2", "--max-time", "24", target])
        payload = json.loads(response)
        if "error" in payload:
            raise ValueError(payload["error"])
        return payload

    total = query(returnCountOnly="true")["count"]
    for offset in range(0, total, 800):
        page = query(outFields=f"{oid},{field}", returnGeometry="true", outSR=4326, orderByFields=oid,
                     resultOffset=offset, resultRecordCount=800)
        points.extend(page["features"])
    ids = [item["attributes"][oid] for item in points]
    if len(points) != total or len(set(ids)) != total:
        raise RuntimeError("Incomplete or repeated census response")
    write_json(local, points)
    print(f"{local.stem}: {len(points)} official records", flush=True)
    return points


def index_roofs(rows):
    records, shapes = [], []
    for row in rows:
        chip = row["chip"]
        path = Path(row["source"]) / "labels" / f"{chip}_post_disaster.json"
        features = json.loads(path.read_bytes())["features"]
        pixel = {b["properties"]["uid"]: b for b in features.get("xy", [])}
        for item in features.get("lng_lat", []):
            uid = item["properties"]["uid"]
            if uid not in pixel:
                continue
            geometry = wkt.loads(item["wkt"])
            if geometry.is_empty or geometry.area <= 0 or not geometry.is_valid:
                continue
            shape = wkt.loads(pixel[uid]["wkt"])
            if shape.is_empty or shape.area <= 0:
                continue
            records.append({"chip": chip, "uid": uid, "bounds": shape.bounds,
                            "xbd": pixel[uid]["properties"].get("subtype", "un-classified")})
            shapes.append(geometry)
    return records, shapes, STRtree(shapes), {id(shape): index for index, shape in enumerate(shapes)}


def containing(point, indexed):
    records, shapes, tree, identities = indexed
    hits = {}
    for value in tree.query(point):
        index = identities[id(value)] if hasattr(value, "covers") else int(value)
        if shapes[index].covers(point):
            row = records[index]
            hits.setdefault(row["uid"], row)
    return hits


def audit(points, training, reserved, destination=OUT, source=BASE, oid="OBJECTID",
          field="TIPO_DE_R_IESGO", names=RISK_NAMES):
    matched = defaultdict(list)
    statuses = Counter()
    census_counts = Counter()
    for feature in points:
        attrs = feature["attributes"]
        raw_risk = str(attrs.get(field) or "").strip().upper()
        risk = names.get(raw_risk, raw_risk or "missing")
        census_counts[risk] += 1
        geom = feature.get("geometry")
        if not geom or not {"x", "y"} <= geom.keys() or not raw_risk or (names and risk not in names.values()):
            continue
        point = Point(geom["x"], geom["y"])
        hits = containing(point, training)
        if len(hits) != 1:
            statuses["no_containment" if not hits else "ambiguous_roof"] += 1
            continue
        if containing(point, reserved):
            statuses["evaluated_region_overlap"] += 1
            continue
        uid, row = next(iter(hits.items()))
        matched[uid].append({**row, "survey_risk": risk, "census_id": attrs[oid]})
    unique = []
    for values in matched.values():
        if len({v["survey_risk"] for v in values}) != 1:
            statuses["conflicting_survey_roof"] += 1
            continue
        unique.append(values[0])
    cross = defaultdict(Counter)
    for row in unique:
        cross[row["survey_risk"]][row["xbd"]] += 1
    result = {"source": source, "census_counts": census_counts, "match_exclusions": statuses,
              "matched_roofs": len(unique), "training_scenes": len({r["chip"] for r in unique}),
              "survey_risk_by_xbd_class": cross}
    write_json(destination / "audit.json", result)
    write_json(destination / "train_matches.json", unique)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def survey_sources():
    yield OUT, BASE, "OBJECTID", "TIPO_DE_R_IESGO", RISK_NAMES
    structural = {"CHOICE0": "structural_damage", "CHOICE1": "no_structural_damage"}
    for i, field in ((1, "field_23"), (2, "field_25")):
        yield OUT / f"unam_{i}", f"{SERVICES}/BrigadasUNAM/MapServer/{i}", "ObjectId", field, structural
    for i in (21, 33, 36, 39, 42):
        yield OUT / f"unosat_{i}", f"{SERVICES}/Unitar/MapServer/{i}", "OBJECTID", "Grouped_Da", None


def load_source(spec):
    path, layer, oid, field, names = spec
    path.mkdir(parents=True, exist_ok=True)
    return fetch(layer, path / "points.json", oid, field)


if __name__ == "__main__":
    manifest = json.loads((ROOT / "data/manifest.json").read_bytes())
    train = index_roofs(manifest["train"])
    reserved = index_roofs(manifest["val"] + manifest["test"])
    sources = list(survey_sources())
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(load_source, spec) for spec in sources]
        for spec, result in zip(sources, futures):
            path, layer, oid, field, names = spec
            audit(result.result(), train, reserved, path, layer, oid, field, names or {})
