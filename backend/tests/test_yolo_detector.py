import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.schemas.inspection import BBox, Detection
from app.services.detection.yolo_detector import YoloDetector, file_fingerprint, to_detections
from app.services.visualization.detections import draw_detections

REPO = Path(__file__).resolve().parents[2]
NAMES = {0: "nut", 1: "washer"}


def test_to_detections_converts_clamps_filters_and_sorts():
    xyxy = np.array([[10, 10, 50, 50], [-5, -5, 30, 30], [90, 70, 130, 100], [20, 20, 20, 40]], dtype=float)
    confs = np.array([0.5, 0.9, 0.7, 0.8])
    cls = np.array([0, 1, 0, 1])
    dets = to_detections(xyxy, confs, cls, NAMES, width=100, height=80)
    assert [d.class_name for d in dets] == ["washer", "nut", "nut"]  # degenerate box dropped
    assert [round(d.confidence, 2) for d in dets] == [0.9, 0.7, 0.5]  # sorted
    assert dets[0].bbox == BBox(x1=0, y1=0, x2=30, y2=30)  # clamped to the image
    assert dets[1].bbox == BBox(x1=90, y1=70, x2=100, y2=80)


def test_to_detections_empty():
    assert to_detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0), NAMES, 100, 80) == []


def test_missing_weights_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        YoloDetector(tmp_path / "nope.pt")


def test_fingerprint_is_stable_and_content_dependent(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"abc")
    b.write_bytes(b"abd")
    assert file_fingerprint(a) == file_fingerprint(a) != file_fingerprint(b)
    assert len(file_fingerprint(a)) == 12


def test_importing_detector_does_not_import_ultralytics():
    code = "import sys, app.services.detection.yolo_detector; assert 'ultralytics' not in sys.modules"
    env = {**os.environ, "PYTHONPATH": str(REPO / "backend")}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr


def test_draw_detections_adds_banner_and_keeps_input():
    img = np.full((100, 200, 3), 30, np.uint8)
    before = img.copy()
    dets = [Detection(class_name="nut", confidence=0.9, bbox=BBox(x1=10, y1=10, x2=50, y2=50))]
    out = draw_detections(img, dets, title="t")
    assert out.shape[1] == 200 and out.shape[0] > 100
    assert np.array_equal(img, before)
    assert draw_detections(img, [], title="").shape == img.shape