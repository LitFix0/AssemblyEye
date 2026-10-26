import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.core.image_io import ImageValidationError, load_image, save_image
from app.schemas.inspection import BBox, Detection, Status
from app.schemas.spec import ProductSpec, Thresholds, load_spec
from app.services.pipeline import run_inspection
from app.services.verification.count_rules import verify_counts
from app.services.visualization.draw import draw_result

REPO = Path(__file__).resolve().parents[2]


def det(cls, conf, x2=50):
    return Detection(class_name=cls, confidence=conf, bbox=BBox(x1=10, y1=10, x2=x2, y2=50))


def normal_parts():
    # order: screw_long, screw_short, nut, nut, washer, washer
    return [det("screw_long", .9), det("screw_short", .9), det("nut", .9),
            det("nut", .9), det("washer", .9), det("washer", .9)]


@pytest.fixture
def spec():
    return ProductSpec(product="t", version="1", thresholds=Thresholds(accept_conf=0.6, review_conf=0.3),
                       required_counts={"screw_long": 1, "screw_short": 1, "nut": 2, "washer": 2})


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "synthetic.png"
    save_image(p, np.full((300, 400, 3), 40, dtype=np.uint8))
    return p


# --- image validation ---
def test_load_valid_image(image_path):
    img = load_image(image_path)
    assert (img.width, img.height) == (400, 300) and img.bgr.shape == (300, 400, 3)


def test_load_rejects_bad_inputs(tmp_path):
    with pytest.raises(ImageValidationError, match="not found"):
        load_image(tmp_path / "nope.png")
    txt = tmp_path / "a.txt"
    txt.write_text("hi")
    with pytest.raises(ImageValidationError, match="unsupported"):
        load_image(txt)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ImageValidationError, match="empty"):
        load_image(empty)
    fake = tmp_path / "fake.png"
    fake.write_text("not an image")
    with pytest.raises(ImageValidationError, match="decoded"):
        load_image(fake)
    tiny = tmp_path / "tiny.png"
    save_image(tiny, np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ImageValidationError, match="dimensions"):
        load_image(tiny)


# --- schemas and spec ---
def test_bbox_and_confidence_validation():
    with pytest.raises(ValidationError):
        BBox(x1=5, y1=5, x2=5, y2=10)
    with pytest.raises(ValidationError):
        det("nut", 1.5)


def test_spec_loads_and_validates():
    s = load_spec(REPO / "configs" / "products" / "screw_bag.yaml")
    assert s.required_counts == {"screw_long": 1, "screw_short": 1, "nut": 2, "washer": 2}
    with pytest.raises(ValidationError):
        Thresholds(accept_conf=0.3, review_conf=0.6)
    with pytest.raises(ValidationError):
        ProductSpec(product="x", version="1", thresholds=Thresholds(accept_conf=.6, review_conf=.3),
                    required_counts={})


# --- count rules ---
def test_normal_passes(spec):
    status, findings = verify_counts(normal_parts(), spec)
    assert status == Status.PASS and findings == []


def test_empty_detections_fail(spec):
    status, findings = verify_counts([], spec)
    assert status == Status.FAIL and {f.code for f in findings} == {"MISSING"} and len(findings) == 4


def test_missing_nut_fails(spec):
    dets = normal_parts()
    dets.pop(2)
    status, findings = verify_counts(dets, spec)
    assert status == Status.FAIL
    assert [(f.code, f.class_name) for f in findings] == [("MISSING", "nut")]


def test_extra_washer_fails(spec):
    status, findings = verify_counts(normal_parts() + [det("washer", .9)], spec)
    assert status == Status.FAIL
    assert [(f.code, f.class_name) for f in findings] == [("EXTRA", "washer")]


def test_low_confidence_nut_goes_to_review(spec):
    dets = normal_parts()
    dets[2] = det("nut", 0.45)
    status, findings = verify_counts(dets, spec)
    assert status == Status.REVIEW and findings[0].code == "UNCERTAIN_PART"


def test_uncertain_extra_goes_to_review(spec):
    status, findings = verify_counts(normal_parts() + [det("nut", 0.4)], spec)
    assert status == Status.REVIEW and findings[0].code == "UNCERTAIN_EXTRA"


def test_below_review_threshold_is_ignored_not_counted(spec):
    dets = normal_parts()
    dets[2] = det("nut", 0.1)
    status, findings = verify_counts(dets, spec)
    assert status == Status.FAIL and findings[0].code == "MISSING"


def test_unexpected_class_goes_to_review(spec):
    status, findings = verify_counts(normal_parts() + [det("bolt", .9)], spec)
    assert status == Status.REVIEW and findings[0].code == "UNEXPECTED_CLASS"


def test_fail_beats_review(spec):
    dets = normal_parts()
    dets.pop(2)
    dets.append(det("washer", 0.4))
    status, findings = verify_counts(dets, spec)
    assert status == Status.FAIL and {f.code for f in findings} == {"MISSING", "UNCERTAIN_EXTRA"}


# --- pipeline, drawing, CLI ---
def test_pipeline_output_format(spec, image_path):
    result, _ = run_inspection(image_path, spec, normal_parts(), "test")
    data = json.loads(result.model_dump_json())
    assert set(data) == {"product", "spec_version", "status", "image", "detections", "findings", "detector"}
    assert data["status"] == "PASS"
    assert data["image"] == {"name": "synthetic.png", "width": 400, "height": 300}
    assert len(data["detections"]) == 6


def test_pipeline_rejects_out_of_bounds_box(spec, image_path):
    with pytest.raises(ValueError, match="outside the image"):
        run_inspection(image_path, spec, [det("nut", .9, x2=500)], "test")


def test_draw_result_adds_banner_and_keeps_input(spec, image_path):
    result, img = run_inspection(image_path, spec, normal_parts(), "test")
    before = img.bgr.copy()
    out = draw_result(img.bgr, result, spec)
    assert out.shape[1] == before.shape[1] and out.shape[0] > before.shape[0]
    assert np.array_equal(img.bgr, before)


def run_cli(*args):
    cmd = [sys.executable, str(REPO / "scripts" / "run_baseline.py"), *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def test_cli_valid_run_writes_json_and_png(tmp_path, image_path):
    fx = tmp_path / "fx.json"
    fx.write_text(json.dumps({"detections": [d.model_dump() for d in normal_parts()]}))
    r = run_cli(image_path, "--detections", fx, "--out-dir", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    jsons = list((tmp_path / "out").glob("*.json"))
    assert len(jsons) == 1 and json.loads(jsons[0].read_text())["status"] == "PASS"
    assert list((tmp_path / "out").glob("*.png"))


def test_cli_invalid_inputs_exit_cleanly(tmp_path, image_path):
    fx = tmp_path / "fx.json"
    fx.write_text(json.dumps({"detections": []}))
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    for target in (tmp_path / "missing.png", bad):
        r = run_cli(target, "--detections", fx, "--out-dir", tmp_path / "out")
        assert r.returncode == 2 and "ERROR" in r.stderr and "Traceback" not in r.stderr