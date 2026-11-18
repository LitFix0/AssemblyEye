"""ST-04: splicing_connectors verification with the real product spec.

Geometry comes from a real annotated normal image (train_005, 1700x850): two connector_2 boxes with the
cable box overlapping each by a few px. Pixels are synthetic: grey background, light-grey connectors,
a thick coloured line for the cable.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.image_io import save_image
from app.schemas.inspection import BBox, Detection, Status
from app.schemas.spec import load_spec
from app.services.pipeline import run_inspection
from app.services.verification.engine import verify_product

SPEC_PATH = Path(__file__).resolve().parents[2] / "configs" / "products" / "splicing_connectors.yaml"
W, H = 1700, 850
LEFT = (189.4, 266.2, 483.6, 500.8)  # connector boxes and cable box from train_005
RIGHT = (1272.4, 298.6, 1575.6, 513.4)
CABLE = (478.2, 385.3, 1283.2, 482.8)
YELLOW, BLUE, RED, GREY = (0, 230, 240), (230, 40, 20), (20, 20, 225), (150, 150, 150)  # BGR


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC_PATH)


def image(cable_bgr=YELLOW, cable=CABLE, transpose=False):
    img = np.full((H, W, 3), 110, dtype=np.uint8)
    for x1, y1, x2, y2 in (LEFT, RIGHT):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (205, 205, 205), -1)
    y = int((cable[1] + cable[3]) / 2)
    cv2.line(img, (int(cable[0]), y), (int(cable[2]), y), cable_bgr, 36)
    return np.ascontiguousarray(img.transpose(1, 0, 2)) if transpose else img


def det(cls, conf, box, transpose=False):
    x1, y1, x2, y2 = (box[1], box[0], box[3], box[2]) if transpose else box
    return Detection(class_name=cls, confidence=conf, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))


def good(conn="connector_2", cable_conf=0.5, conn_conf=0.9, cable=CABLE):
    return [det(conn, conn_conf, LEFT), det(conn, conn_conf - 0.02, RIGHT), det("cable", cable_conf, cable)]


def codes(findings):
    return sorted(f.code for f in findings)


# ---- correct assemblies -----------------------------------------------------
@pytest.mark.parametrize("conn,bgr", [("connector_2", YELLOW), ("connector_3", BLUE), ("connector_5", RED)])
def test_correct_assembly_passes_for_each_type(spec, conn, bgr):
    status, findings = verify_product(good(conn), spec, image(bgr))
    assert status == Status.PASS and findings == []


def test_cable_at_realistic_confidence_is_confident_with_its_own_threshold(spec):
    # cable 0.45 is below the global accept (0.6) but above the cable accept (0.4)
    assert verify_product(good(cable_conf=0.45), spec, image())[0] == Status.PASS


# ---- missing ----------------------------------------------------------------
def test_missing_cable_fails(spec):
    status, findings = verify_product(good()[:2], spec, image())
    assert status == Status.FAIL and [(f.code, f.class_name) for f in findings] == [("MISSING", "cable")]


def test_missing_connector_fails_and_stops_geometry(spec):
    status, findings = verify_product([good()[0], good()[2]], spec, image())
    assert status == Status.FAIL and codes(findings) == ["MISSING"] and findings[0].class_name == "connector"


def test_cable_below_review_threshold_counts_as_missing(spec):
    status, findings = verify_product(good(cable_conf=0.1), spec, image())
    assert status == Status.FAIL and codes(findings) == ["MISSING"]


# ---- misplaced / not connected ---------------------------------------------
def test_cable_far_from_the_connectors_fails_on_both_sides(spec):
    low = (CABLE[0], CABLE[1] + 330, CABLE[2], CABLE[3] + 330)
    status, findings = verify_product(good(cable=low), spec, image(cable=low))
    assert status == Status.FAIL
    assert codes(findings) == ["CABLE_NOT_CONNECTED", "CABLE_NOT_CONNECTED"]
    assert {f.evidence["side"] for f in findings} == {"left", "right"}
    assert all(f.evidence["gap_px"] > f.evidence["tolerance_px"] for f in findings)


def test_cable_too_short_fails_on_the_open_side_only(spec):
    short = (CABLE[0], CABLE[1], 900.0, CABLE[3])
    status, findings = verify_product(good(cable=short), spec, image(cable=short))
    assert status == Status.FAIL and codes(findings) == ["CABLE_NOT_CONNECTED"]
    f = findings[0]
    assert f.evidence["side"] == "right" and "right connector" in f.message and f.evidence["gap_px"] > 300


def test_cable_sticking_out_past_a_connector_fails(spec):
    long_ = (60.0, CABLE[1], CABLE[2], CABLE[3])  # starts 130 px before the left connector's far edge
    status, findings = verify_product(good(cable=long_), spec, image(cable=long_))
    assert status == Status.FAIL and codes(findings) == ["CABLE_MISPLACED"]
    assert findings[0].evidence["side"] == "left" and findings[0].evidence["overshoot_px"] > 100


def test_small_box_wobble_is_tolerated(spec):
    wobble = (CABLE[0] + 15, CABLE[1] - 8, CABLE[2] - 12, CABLE[3] + 6)
    assert verify_product(good(cable=wobble), spec, image())[0] == Status.PASS


def test_vertical_layout_uses_the_other_axis(spec):
    # same assembly rotated 90 degrees (x and y swapped): still PASS; a gap is reported as upper/lower
    dets = [det(d.class_name, d.confidence, (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2), transpose=True)
            for d in good()]
    assert verify_product(dets, spec, image(transpose=True))[0] == Status.PASS
    short = (CABLE[0], CABLE[1], 900.0, CABLE[3])
    dets = [det(d.class_name, d.confidence, (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2), transpose=True)
            for d in good(cable=short)]
    status, findings = verify_product(dets, spec, image(cable=short, transpose=True))
    assert status == Status.FAIL and findings[0].evidence["side"] == "lower"


# ---- duplicates / extra -----------------------------------------------------
def test_duplicate_cable_box_is_review_and_counted_once(spec):
    dup = (CABLE[0] + 4, CABLE[1] + 2, CABLE[2] - 3, CABLE[3] - 2)
    status, findings = verify_product(good() + [det("cable", 0.42, dup)], spec, image())
    assert status == Status.REVIEW and codes(findings) == ["DUPLICATE_DETECTION"]


def test_second_distinct_cable_fails_as_extra(spec):
    other = (CABLE[0], CABLE[1] + 200, CABLE[2], CABLE[3] + 200)
    status, findings = verify_product(good() + [det("cable", 0.6, other)], spec, image())
    assert status == Status.FAIL and [(f.code, f.class_name) for f in findings] == [("EXTRA", "cable")]


def test_third_connector_fails_as_extra(spec):
    third = (700.0, 600.0, 900.0, 780.0)
    status, findings = verify_product(good() + [det("connector_2", 0.9, third)], spec, image())
    assert status == Status.FAIL and codes(findings) == ["EXTRA"]


# ---- wrong class / wrong colour --------------------------------------------
def test_mixed_connector_types_fail(spec):
    dets = [det("connector_2", 0.9, LEFT), det("connector_3", 0.9, RIGHT), det("cable", 0.5, CABLE)]
    status, findings = verify_product(dets, spec, image())
    assert status == Status.FAIL and codes(findings) == ["CONNECTOR_TYPE_MISMATCH"]
    assert "connector_2 and connector_3" in findings[0].message


def test_wrong_cable_colour_fails_with_evidence(spec):
    status, findings = verify_product(good("connector_2"), spec, image(BLUE))
    assert status == Status.FAIL and codes(findings) == ["CABLE_COLOR_MISMATCH"]
    f = findings[0]
    assert "looks blue" in f.message and "needs yellow" in f.message
    assert f.evidence["expected"] == "yellow" and f.evidence["observed"] == "blue" and f.evidence["share"] > 0.9


def test_unreadable_cable_colour_is_review(spec):
    status, findings = verify_product(good(), spec, image(GREY))
    assert status == Status.REVIEW and codes(findings) == ["CABLE_COLOR_UNKNOWN"]
    assert "by eye" in findings[0].message and findings[0].evidence["observed"] is None


def test_unclear_wrong_colour_is_review_not_fail(spec):
    # mostly blue (a clear winner, but under confident_share 0.7) on a yellow-needing connector
    img = image(YELLOW)
    y = int((CABLE[1] + CABLE[3]) / 2)
    cv2.line(img, (int(CABLE[0] + 100), y), (int(CABLE[0] + 100 + 500), y), BLUE, 36)
    status, findings = verify_product(good(), spec, img)
    f = [x for x in findings if x.code == "CABLE_COLOR_MISMATCH"]
    assert f and f[0].status == Status.REVIEW and status == Status.REVIEW
    assert spec.splicing.color.min_share < f[0].evidence["share"] < spec.splicing.color.confident_share


def test_connector_classes_competing_on_one_object_is_review(spec):
    dets = good() + [det("connector_3", 0.5, LEFT)]
    status, findings = verify_product(dets, spec, image())
    assert status == Status.REVIEW and codes(findings) == ["COMPETING_CLASSES"]


# ---- uncertain --------------------------------------------------------------
def test_uncertain_cable_is_review_and_geometry_is_not_judged(spec):
    status, findings = verify_product(good(cable_conf=0.3), spec, image())
    assert status == Status.REVIEW and codes(findings) == ["UNCERTAIN_PART"]


def test_uncertain_connector_is_review(spec):
    dets = [det("connector_2", 0.9, LEFT), det("connector_2", 0.45, RIGHT), det("cable", 0.5, CABLE)]
    status, findings = verify_product(dets, spec, image())
    assert status == Status.REVIEW and codes(findings) == ["UNCERTAIN_PART"]


def test_unexpected_class_is_review(spec):
    status, findings = verify_product(good() + [det("nut", 0.9, (10, 10, 60, 60))], spec, image())
    assert status == Status.REVIEW and codes(findings) == ["UNEXPECTED_CLASS"]


def test_nothing_uncertain_ever_passes(spec):
    for dets in (good(cable_conf=0.39), good(conn_conf=0.55), good(conn_conf=0.59)):
        assert verify_product(dets, spec, image())[0] != Status.PASS


# ---- engine and pipeline ----------------------------------------------------
def test_splicing_requires_the_image(spec):
    with pytest.raises(ValueError, match="needs the image"):
        verify_product(good(), spec, None)


def test_deterministic_and_order_independent(spec):
    dets = good(cable_conf=0.45) + [det("cable", 0.3, (400, 700, 900, 760))]
    img = image()
    a = verify_product(dets, spec, img)
    b = verify_product(list(reversed(dets)), spec, img)
    assert a[0] == b[0] and sorted(codes(a[1])) == sorted(codes(b[1]))
    assert verify_product(dets, spec, img) == a


def test_pipeline_runs_splicing_and_result_is_json(spec, tmp_path):
    p = tmp_path / "splice.png"
    save_image(p, image(BLUE))
    result, _ = run_inspection(p, spec, good("connector_2"), "test")
    data = result.model_dump(mode="json")
    assert data["status"] == "FAIL" and data["product"] == "splicing_connectors"
    assert data["findings"][0]["code"] == "CABLE_COLOR_MISMATCH" and data["findings"][0]["evidence"]["observed"] == "blue"


# ---- spec validation --------------------------------------------------------
def test_spec_loads_and_validates(spec):
    assert spec.splicing.cable_colors == {"connector_2": "yellow", "connector_3": "blue", "connector_5": "red"}
    assert spec.thresholds_for("cable").accept_conf == 0.4 and spec.thresholds_for("connector_2").accept_conf == 0.6
    assert spec.min_review_conf == 0.2


def test_bad_splicing_specs_are_rejected(spec):
    base = spec.splicing.model_dump()
    from app.schemas.spec import SplicingRules
    for change in ({"cable_colors": {"connector_2": "yellow"}},
                   {"cable_colors": {**base["cable_colors"], "connector_5": "purple"}},
                   {"cable_class": "connector_2"},
                   {"end_tolerance": 0.0}):
        with pytest.raises(ValueError):
            SplicingRules.model_validate({**base, **change})

# ---- weak duplicate cable box and minimum cable length -----------------------
def test_weak_second_box_on_the_same_cable_is_not_an_extra(spec):
    weak = (CABLE[0], CABLE[1], CABLE[0] + 0.55 * (CABLE[2] - CABLE[0]), CABLE[3])  # IoU 0.55 with the cable
    status, findings = verify_product(good() + [det("cable", 0.30, weak)], spec, image())
    assert status == Status.PASS and findings == []


def test_weak_box_far_from_the_cable_is_still_flagged(spec):
    elsewhere = (200, 20, 700, 120)  # no overlap with the cable
    status, findings = verify_product(good() + [det("cable", 0.30, elsewhere)], spec, image())
    assert status == Status.REVIEW and codes(findings) == ["UNCERTAIN_EXTRA"]


def test_confident_second_cable_still_fails_even_if_overlapping(spec):
    same = (CABLE[0], CABLE[1], CABLE[0] + 0.55 * (CABLE[2] - CABLE[0]), CABLE[3])  # IoU 0.55
    status, findings = verify_product(good() + [det("cable", 0.55, same)], spec, image())
    assert status == Status.FAIL and "EXTRA" in codes(findings)


def test_short_cable_fails(spec):
    short = (CABLE[0], CABLE[1], CABLE[0] + 170, CABLE[3])
    _, findings = verify_product(good(cable=short), spec, image(cable=short))
    assert "CABLE_TOO_SHORT" in codes(findings)
    assert any(f.status == Status.FAIL for f in findings if f.code == "CABLE_TOO_SHORT")


def test_normal_length_cable_has_no_length_finding(spec):
    _, findings = verify_product(good(), spec, image())
    assert "CABLE_TOO_SHORT" not in codes(findings)