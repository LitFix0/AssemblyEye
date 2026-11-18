"""The eval_defects script end to end, with a fake detector and a tiny fake LOCO folder.

No weights or dataset are needed: this checks the plumbing (spec choice, splicing support, JSON output,
the 'suppress' field), not detector quality."""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.schemas.inspection import BBox, Detection

REPO = Path(__file__).resolve().parents[2]


def load_script():
    spec = importlib.util.spec_from_file_location("eval_defects_script", REPO / "scripts" / "eval_defects.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_png(path, bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def build_category(root, category, n_test_good=2):
    base = root / category
    write_png(base / "train" / "good" / "000.png", np.full((200, 300, 3), 100, np.uint8))
    rels, groups, strata = [], {}, {}
    for i in range(n_test_good):
        rel = f"test/good/{i:03d}.png"
        write_png(base / rel, np.full((200, 300, 3), 100, np.uint8))
        rels.append(rel)
        groups[rel], strata[rel] = "normal", "good"
    split = {"assignment": {r: "dev" for r in rels}, "groups": groups, "strata": strata}
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))
    return split_path


class FakeDetector:
    class_names = ["screw_long", "screw_short", "nut", "washer"]
    name = "fake"

    def __init__(self, *a, **k):
        pass

    def detect(self, bgr):
        d = lambda c, x: Detection(class_name=c, confidence=0.9, bbox=BBox(x1=x, y1=10, x2=x + 30, y2=50))
        return [d(c, 10 + 40 * i) for i, c in enumerate(["screw_long", "screw_short", "nut", "nut",
                                                        "washer", "washer"])]


def test_screw_bag_run_writes_suppress_field(tmp_path, monkeypatch, capsys):
    mod = load_script()
    monkeypatch.setattr(mod, "YoloDetector", FakeDetector)
    monkeypatch.chdir(tmp_path)
    split = build_category(tmp_path / "root", "screw_bag")
    rc = mod.main(["--category", "screw_bag", "--weights", "w.pt", "--imgsz", "640",
                   "--root", str(tmp_path / "root"), "--split-file", str(split),
                   "--spec", str(REPO / "configs/products/screw_bag.yaml")])
    assert rc == 0
    out = list((tmp_path / "reports").glob("eval_defects_screw_bag_*.json"))
    assert len(out) == 1
    data = json.loads(out[0].read_text())
    assert data["suppress"] is True and data["groups"]["normal"]["PASS"] == 2
    assert data["rows"][0]["messages"] == []

    rc = mod.main(["--category", "screw_bag", "--weights", "w.pt", "--imgsz", "640", "--no-suppress",
                   "--root", str(tmp_path / "root"), "--split-file", str(split),
                   "--spec", str(REPO / "configs/products/screw_bag.yaml")])
    assert rc == 0
    nosup = json.loads(next((tmp_path / "reports").glob("*_nosup.json")).read_text())
    assert nosup["suppress"] is False


def test_splicing_run_uses_pixels_and_flags_a_normal_with_no_cable_colour(tmp_path, monkeypatch, capsys):
    mod = load_script()

    class SplicingFake:
        class_names = ["connector_2", "connector_3", "connector_5", "cable"]
        name = "fake-splicing"

        def __init__(self, *a, **k):
            pass

        def detect(self, bgr):  # 300x200 image: two connectors and a cable between them
            d = lambda c, conf, b: Detection(class_name=c, confidence=conf, bbox=BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
            return [d("connector_2", 0.9, (10, 60, 70, 140)), d("connector_2", 0.9, (230, 60, 290, 140)),
                    d("cable", 0.5, (65, 90, 235, 110))]

    monkeypatch.setattr(mod, "YoloDetector", SplicingFake)
    monkeypatch.chdir(tmp_path)
    split = build_category(tmp_path / "root", "splicing_connectors")
    rc = mod.main(["--category", "splicing_connectors", "--weights", "w.pt", "--imgsz", "960",
                   "--root", str(tmp_path / "root"), "--split-file", str(split),
                   "--spec", str(REPO / "configs/products/splicing_connectors.yaml")])
    assert rc == 0
    data = json.loads(next((tmp_path / "reports").glob("eval_defects_splicing_connectors_*.json")).read_text())
    # the fake image is plain grey, so the cable colour cannot be read: REVIEW, never a silent PASS
    assert data["suppress"] is False and data["spec_version"].startswith("0.2")
    assert data["groups"]["normal"]["REVIEW"] == 2
    assert data["rows"][0]["findings"] == ["CABLE_COLOR_UNKNOWN:cable"]
    assert "why normal images were flagged" in capsys.readouterr().out


def test_spec_category_mismatch_is_a_clean_error(tmp_path, monkeypatch, capsys):
    mod = load_script()
    monkeypatch.setattr(mod, "YoloDetector", FakeDetector)
    monkeypatch.chdir(tmp_path)
    split = build_category(tmp_path / "root", "splicing_connectors")
    rc = mod.main(["--category", "splicing_connectors", "--weights", "w.pt", "--imgsz", "960",
                   "--root", str(tmp_path / "root"), "--split-file", str(split),
                   "--spec", str(REPO / "configs/products/screw_bag.yaml")])
    assert rc == 2 and "does not match category" in capsys.readouterr().err
