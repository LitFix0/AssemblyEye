import csv
import json

import numpy as np
import pytest
import yaml
from PIL import Image

from app.datasets.labelme_io import AnnotationError, Box, parse_labelme, to_yolo_line
from app.datasets.selection import LeakageError
from app.datasets.yolo_export import (assert_manifest_clean, composition_issues, export_yolo,
                                      read_yolo_labels, split_train_pool)

CLASSES = ["a", "b"]
W, H = 100, 80
GOOD = [
    {"label": "a", "points": [[10, 20], [50, 60]], "shape_type": "rectangle"},
    {"label": "b", "points": [[60, 10], [90, 70]], "shape_type": "rectangle"},
]


def shape(label, pts, shape_type="rectangle"):
    return {"label": label, "points": pts, "group_id": None, "shape_type": shape_type, "flags": {}}


def write_json(path, shapes, w=W, h=H):
    path.write_text(json.dumps({"version": "test", "flags": {}, "shapes": shapes, "imagePath": "x.png",
                                "imageData": None, "imageHeight": h, "imageWidth": w}))


# --- parsing ---
def test_parse_normalizes_reversed_points_and_clips(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, [shape("a", [[50, 60], [10, 20]]), shape("b", [[-3, 0], [40, 83]])])
    boxes = parse_labelme(p, CLASSES, image_size=(W, H))
    assert boxes[0] == Box("a", 10, 20, 50, 60)
    assert boxes[1] == Box("b", 0, 0, 40, 80)


@pytest.mark.parametrize("shapes, match", [
    ([shape("zzz", [[1, 1], [20, 20]])], "unknown label"),
    ([shape("a", [[1, 1], [20, 20]], "line")], "not supported"),
    ([], "empty annotation"),
    ([shape("a", [[10, 10], [120, 40]])], "outside the image"),
    ([shape("a", [[10, 10], [11, 40]])], "too small"),
    ([shape("a", [[10, 10]])], "at least 2 points"),
])
def test_parse_rejects_bad_shapes(tmp_path, shapes, match):
    p = tmp_path / "x.json"
    write_json(p, shapes)
    with pytest.raises(AnnotationError, match=match):
        parse_labelme(p, CLASSES)


def test_parse_rejects_bad_files(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, GOOD)
    with pytest.raises(AnnotationError, match="JSON says"):
        parse_labelme(p, CLASSES, image_size=(200, 80))
    p.write_text("{")
    with pytest.raises(AnnotationError, match="cannot read JSON"):
        parse_labelme(p, CLASSES)
    p.write_text("{}")
    with pytest.raises(AnnotationError, match="missing or invalid field"):
        parse_labelme(p, CLASSES)

def test_polygon_and_circle_convert_to_bounding_boxes(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, [
        shape("a", [[10, 20], [50, 25], [45, 60], [12, 55]], "polygon"),
        shape("b", [[70, 40], [80, 40]], "circle"),  # centre (70, 40), radius 10
    ])
    a, b = parse_labelme(p, CLASSES, image_size=(W, H))
    assert (a.x1, a.y1, a.x2, a.y2) == (10, 20, 50, 60) and a.source_shape == "polygon"
    assert (b.x1, b.y1, b.x2, b.y2) == (60, 30, 80, 50) and b.source_shape == "circle"


def test_degenerate_polygon_and_circle_rejected(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, [shape("a", [[10, 10], [30, 30]], "polygon")])
    with pytest.raises(AnnotationError, match="at least 3 points"):
        parse_labelme(p, CLASSES)
    write_json(p, [shape("a", [[10, 10], [10, 10]], "circle")])
    with pytest.raises(AnnotationError, match="too small"):
        parse_labelme(p, CLASSES)
    write_json(p, [shape("a", [[10, 10]], "circle")])
    with pytest.raises(AnnotationError, match="2 points"):
        parse_labelme(p, CLASSES)


def test_yolo_line_math():
    assert to_yolo_line(Box("a", 10, 20, 50, 60), 0, 100, 80) == "0 0.300000 0.500000 0.400000 0.500000"


# --- split and guards ---
def test_split_is_deterministic_disjoint_and_never_empty():
    files = [f"train_{i:03d}.png" for i in range(16)]
    tr, va = split_train_pool(files, seed=1)
    assert (tr, va) == split_train_pool(files, seed=1)
    assert not set(tr) & set(va) and sorted(tr + va) == files and len(va) == 3
    tr2, va2 = split_train_pool(files[:2], seed=1)
    assert len(tr2) == 1 and len(va2) == 1
    assert split_train_pool(files[:1])[1] == []


def test_manifest_guard_rejects_test_and_role_mismatch():
    ok = {"file": "train_001.png", "role": "train_pool", "source": "train/good/001.png"}
    assert_manifest_clean([ok])
    for bad in ({**ok, "source": "test/good/001.png"}, {**ok, "role": "holdout"}, {**ok, "role": "mystery"}):
        with pytest.raises(LeakageError):
            assert_manifest_clean([bad])


def test_composition_issues():
    spec = {"screw_long": 1, "screw_short": 1, "nut": 2, "washer": 2}
    ok = {"screw_long": 1, "screw_short": 1, "nut": 2, "washer": 2}
    assert composition_issues("screw_bag", ok, spec) == []
    assert composition_issues("screw_bag", {**ok, "nut": 3}, spec) == ["nut: expected 2, labelled 3"]
    assert composition_issues("splicing_connectors", {"connector_5": 2, "cable": 1}) == []
    assert composition_issues("splicing_connectors", {"connector_5": 1, "connector_3": 1, "cable": 1})
    assert composition_issues("splicing_connectors", {"connector_2": 2})


# --- export ---
@pytest.fixture
def cat_dir(tmp_path):
    d = tmp_path / "ann" / "cat"
    (d / "images").mkdir(parents=True)
    (d / "labels_json").mkdir()
    rows = [(f"train_{i:03d}.png", "train_pool", f"train/good/{i:03d}.png") for i in range(8)]
    rows += [(f"validation_{i:03d}.png", "holdout", f"validation/good/{i:03d}.png") for i in range(2)]
    for name, _, _ in rows:
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(d / "images" / name)
    with (d / "manifest.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "role", "source"])
        wr.writerows(rows)
    return d


def label_all(d, skip=()):
    for img in sorted((d / "images").glob("*.png")):
        if img.name not in skip:
            write_json(d / "labels_json" / f"{img.stem}.json", GOOD)


def test_export_end_to_end(cat_dir, tmp_path):
    label_all(cat_dir, skip={"train_007.png", "validation_001.png"})
    out = tmp_path / "out"
    rep = export_yolo(cat_dir, CLASSES, out, seed=3)
    assert rep["images"] == {"train": 6, "val": 1, "holdout": 1}
    assert rep["unlabelled"] == ["train_007.png", "validation_001.png"]
    assert rep["boxes_per_class"] == {"a": 8, "b": 8}
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["names"] == {0: "a", 1: "b"} and cfg["test"] == "images/holdout" and "path" not in cfg
    assert len(list((out / "images" / "holdout").glob("*.png"))) == 1
    labels = list((out / "labels").rglob("*.txt"))
    assert len(labels) == 8
    for lf in labels:  # round trip: pixel boxes survive the conversion
        got = read_yolo_labels(lf, W, H)
        assert [g[0] for g in got] == [0, 1]
        assert got[0][1:] == pytest.approx((10, 20, 50, 60), abs=0.01)
        assert got[1][1:] == pytest.approx((60, 10, 90, 70), abs=0.01)


def test_export_is_repeatable_and_clears_stale_files(cat_dir, tmp_path):
    label_all(cat_dir)
    out = tmp_path / "out"
    first = export_yolo(cat_dir, CLASSES, out)
    (out / "images" / "train" / "stale.png").write_bytes(b"x")
    assert export_yolo(cat_dir, CLASSES, out) == first
    assert not (out / "images" / "train" / "stale.png").exists()


def test_export_refuses_foreign_directory(cat_dir, tmp_path):
    label_all(cat_dir)
    out = tmp_path / "precious"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="refusing"):
        export_yolo(cat_dir, CLASSES, out)
    assert (out / "notes.txt").exists()


def test_export_strict_and_error_aggregation(cat_dir, tmp_path):
    label_all(cat_dir, skip={"train_000.png"})
    with pytest.raises(ValueError, match="unlabelled"):
        export_yolo(cat_dir, CLASSES, tmp_path / "o1", strict=True)
    for stem in ("train_001", "train_002"):
        write_json(cat_dir / "labels_json" / f"{stem}.json", [shape("zzz", [[1, 1], [20, 20]])])
    with pytest.raises(AnnotationError, match="2 annotation problem") as e:
        export_yolo(cat_dir, CLASSES, tmp_path / "o2")
    assert "train_001" in str(e.value) and "train_002" in str(e.value)


def test_export_rejects_leaky_manifest(cat_dir, tmp_path):
    label_all(cat_dir)
    text = (cat_dir / "manifest.csv").read_text().replace("train/good/000.png", "test/good/000.png")
    (cat_dir / "manifest.csv").write_text(text)
    with pytest.raises(LeakageError):
        export_yolo(cat_dir, CLASSES, tmp_path / "out")

def test_unreviewed_drafts_are_skipped_and_reported(cat_dir, tmp_path):
    label_all(cat_dir)
    p = cat_dir / "labels_json" / "train_000.json"
    data = json.loads(p.read_text())
    data["flags"] = {"needs_review": True}
    p.write_text(json.dumps(data))
    rep = export_yolo(cat_dir, CLASSES, tmp_path / "out", seed=3)
    assert rep["unreviewed"] == ["train_000.png"]
    assert rep["images"]["train"] + rep["images"]["val"] == 7  # 8 in the pool, minus the draft
    data["flags"] = {"needs_review": False}
    p.write_text(json.dumps(data))
    assert export_yolo(cat_dir, CLASSES, tmp_path / "out2", seed=3)["unreviewed"] == []