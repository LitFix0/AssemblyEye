import csv
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from app.datasets.loco import (DatasetLayoutError, defect_values, index_category,
                               load_defect_names, summarize)
from app.datasets.prepare import prepare_batch
from app.datasets.selection import LeakageError, assert_no_leakage, select_for_annotation

CLASSES = ["a", "b"]


def _png(path, value=0, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0]), value, dtype=np.uint8)).save(path)


@pytest.fixture
def loco(tmp_path):
    base = tmp_path / "raw" / "cat"
    for i in range(10):
        _png(base / "train" / "good" / f"{i:03d}.png")
    for i in range(4):
        _png(base / "validation" / "good" / f"{i:03d}.png")
    for i in range(3):
        _png(base / "test" / "good" / f"{i:03d}.png")
    for i in range(2):
        _png(base / "test" / "logical_anomalies" / f"{i:03d}.png")
        _png(base / "ground_truth" / "logical_anomalies" / f"{i:03d}" / "000.png", value=255)
    _png(base / "ground_truth" / "logical_anomalies" / "001" / "001.png", value=254)  # 2nd defect
    _png(base / "test" / "structural_anomalies" / "000.png")
    _png(base / "ground_truth" / "structural_anomalies" / "000" / "000.png", value=236)
    (base / "defects_config.json").write_text(json.dumps([
        {"defect_name": "d255", "pixel_value": 255},
        {"defect_name": "d254", "pixel_value": 254},
        {"defect_name": "d236", "pixel_value": 236},
    ]))
    return tmp_path / "raw"


def test_index_counts_and_masks(loco):
    s = index_category(loco, "cat")
    assert summarize(s) == {"test/good": 3, "test/logical_anomalies": 2,
                            "test/structural_anomalies": 1, "train/good": 10,
                            "validation/good": 4}
    logical = {x.image_path.stem: x for x in s if x.label == "logical_anomalies"}
    assert len(logical["000"].mask_paths) == 1 and len(logical["001"].mask_paths) == 2
    assert all(x.mask_paths == () for x in s if x.is_normal)


def test_defect_values_and_names(loco):
    s = {x.rel_path: x for x in index_category(loco, "cat")}
    assert defect_values(s["test/logical_anomalies/001.png"]) == [254, 255]
    assert defect_values(s["train/good/000.png"]) == []
    assert load_defect_names(loco / "cat")[255] == "d255"


def test_layout_errors(loco):
    with pytest.raises(DatasetLayoutError, match="not found"):
        index_category(loco, "nope")
    shutil.rmtree(loco / "cat" / "ground_truth" / "logical_anomalies" / "000")
    with pytest.raises(DatasetLayoutError, match="no ground-truth mask"):
        index_category(loco, "cat")


def test_anomalies_in_train_rejected(loco):
    _png(loco / "cat" / "train" / "logical_anomalies" / "x.png")
    with pytest.raises(DatasetLayoutError, match="unexpected"):
        index_category(loco, "cat")


def test_selection_deterministic_prefix_and_eligible(loco):
    s = index_category(loco, "cat")
    a = select_for_annotation(s, 6, 2, seed=1)
    b = select_for_annotation(s, 6, 2, seed=1)
    assert [x.rel_path for x, _ in a] == [x.rel_path for x, _ in b]
    small = select_for_annotation(s, 3, 1, seed=1)
    big = select_for_annotation(s, 8, 3, seed=1)
    assert {(x.rel_path, r) for x, r in small} <= {(x.rel_path, r) for x, r in big}
    for smp, role in big:
        assert smp.is_normal and smp.split != "test"
        assert (role == "train_pool") == (smp.split == "train")
    assert len({x.rel_path for x, _ in big}) == len(big)


def test_seed_changes_selection(loco):
    s = index_category(loco, "cat")
    base = [x.rel_path for x, _ in select_for_annotation(s, 5, 2, seed=1)]
    others = [[x.rel_path for x, _ in select_for_annotation(s, 5, 2, seed=k)] for k in range(2, 8)]
    assert any(o != base for o in others)


def test_selection_too_many(loco):
    with pytest.raises(ValueError, match="only"):
        select_for_annotation(index_category(loco, "cat"), 11, 0)


def test_leakage_guard_rejects_test_and_anomalous(loco):
    s = index_category(loco, "cat")
    test_good = next(x for x in s if x.split == "test" and x.is_normal)
    anomalous = next(x for x in s if not x.is_normal)
    for bad in (test_good, anomalous):
        with pytest.raises(LeakageError):
            assert_no_leakage([(bad, "train_pool")])


def test_prepare_copies_and_writes_manifest(loco, tmp_path):
    out = tmp_path / "ann"
    rows = prepare_batch(loco, "cat", CLASSES, 4, 2, out, seed=7)
    cat = out / "cat"
    assert len(rows) == 6
    assert sorted(p.name for p in (cat / "images").glob("*.png")) == sorted(r["file"] for r in rows)
    assert all(r["file"].startswith(("train_", "validation_")) for r in rows)
    assert (cat / "classes.txt").read_text().split() == CLASSES
    with (cat / "manifest.csv").open(newline="") as f:
        assert list(csv.DictReader(f)) == rows


def test_prepare_extends_and_refuses_shrink(loco, tmp_path):
    out = tmp_path / "ann"
    first = prepare_batch(loco, "cat", CLASSES, 3, 1, out, seed=7)
    second = prepare_batch(loco, "cat", CLASSES, 6, 2, out, seed=7)
    assert {r["file"] for r in first} <= {r["file"] for r in second}
    assert prepare_batch(loco, "cat", CLASSES, 6, 2, out, seed=7) == second  # idempotent
    with pytest.raises(ValueError, match="existing manifest"):
        prepare_batch(loco, "cat", CLASSES, 2, 1, out, seed=7)