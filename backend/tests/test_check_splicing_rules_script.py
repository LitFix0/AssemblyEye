import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
W, H = 600, 300
LEFT, RIGHT, CABLE = (20, 80, 140, 220), (460, 80, 580, 220), (135, 130, 465, 170)


def shape(label, box):
    x1, y1, x2, y2 = box
    return {"label": label, "points": [[x1, y1], [x2, y2]], "shape_type": "rectangle",
            "group_id": None, "description": "", "flags": {}}


def make_annotation_dir(root: Path, cable_bgr):
    cat = root / "splicing_connectors"
    (cat / "images").mkdir(parents=True)
    (cat / "labels_json").mkdir()
    img = np.full((H, W, 3), 110, dtype=np.uint8)
    for b in (LEFT, RIGHT):
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (205, 205, 205), -1)
    cv2.line(img, (CABLE[0], 150), (CABLE[2], 150), cable_bgr, 30)
    cv2.imwrite(str(cat / "images" / "train_001.png"), img)
    (cat / "classes.txt").write_text("connector_2\nconnector_3\nconnector_5\ncable\n")
    (cat / "manifest.csv").write_text("file,role,source\ntrain_001.png,train,train/good/001.png\n"
                                      "train_002.png,train,train/good/002.png\n")
    label = {"version": "7.9.1", "flags": {"needs_review": False}, "imageWidth": W, "imageHeight": H,
             "shapes": [shape("connector_2", LEFT), shape("connector_2", RIGHT), shape("cable", CABLE)]}
    (cat / "labels_json" / "train_001.json").write_text(json.dumps(label))


def run(root):
    cmd = [sys.executable, str(REPO / "scripts" / "check_splicing_rules.py"), "--annotation-dir", str(root),
           "--spec", str(REPO / "configs" / "products" / "splicing_connectors.yaml")]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def test_correct_yellow_cable_passes(tmp_path):
    make_annotation_dir(tmp_path, (0, 230, 240))
    r = run(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "train_001.png" in r.stdout and "'PASS': 1" in r.stdout
    assert "yellow 100%" in r.stdout and "skipped train_002.png: no label file" in r.stdout


def test_wrong_colour_is_reported_with_the_reason(tmp_path):
    make_annotation_dir(tmp_path, (230, 40, 20))  # blue cable on connector_2
    r = run(tmp_path)
    assert r.returncode == 0 and "'FAIL': 1" in r.stdout
    assert "CABLE_COLOR_MISMATCH" in r.stdout and "needs yellow" in r.stdout


def test_missing_manifest_is_a_clean_error(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 2 and "ERROR" in r.stderr and "Traceback" not in r.stderr
