"""ST-04: screw_bag verification with the real product spec.

screw_bag is a count product: positions are free, so 'misplaced' does not apply (one test documents
that). The long/short screw confusion is the main known detector error, so it gets REVIEW, not FAIL.
"""
from pathlib import Path

import pytest

from app.schemas.inspection import BBox, Detection, Status
from app.schemas.spec import ProductSpec, Thresholds, load_spec
from app.services.verification.count_rules import verify_counts
from app.services.verification.engine import verify_product

SPEC_PATH = Path(__file__).resolve().parents[2] / "configs" / "products" / "screw_bag.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC_PATH)


def det(cls, conf, x1, y1=10, w=40, h=40):
    return Detection(class_name=cls, confidence=conf, bbox=BBox(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h))


ORDER = ["screw_long", "screw_short", "nut", "nut", "washer", "washer"]


def parts(classes=ORDER, conf=0.9):
    return [det(c, conf, x1=10 + 60 * i) for i, c in enumerate(classes)]


def codes(findings):
    return sorted(f.code for f in findings)


# ---- the six required cases -------------------------------------------------
def test_correct_assembly_passes(spec):
    status, findings = verify_product(parts(), spec)
    assert status == Status.PASS and findings == []


def test_missing_part_fails(spec):
    status, findings = verify_product(parts(["screw_long", "screw_short", "nut", "washer", "washer"]), spec)
    assert status == Status.FAIL
    assert [(f.code, f.class_name) for f in findings] == [("MISSING", "nut")]


def test_misplaced_part_is_not_a_screw_bag_rule(spec):
    # Free-position product: parts anywhere in the bag are fine, only counts matter.
    moved = [det(c, 0.9, x1=700 - 60 * i, y1=500 + 30 * (i % 2)) for i, c in enumerate(ORDER)]
    assert verify_product(moved, spec)[0] == Status.PASS


def test_duplicate_detection_is_review_not_extra_fail(spec):
    dets = parts() + [det("nut", 0.7, x1=10 + 60 * 2 + 1)]  # same nut, boxed twice (IoU ~ 0.95)
    status, findings = verify_product(dets, spec)
    assert status == Status.REVIEW and codes(findings) == ["DUPLICATE_DETECTION"]
    assert findings[0].evidence["iou"] >= 0.7


def test_real_extra_part_still_fails(spec):
    status, findings = verify_product(parts() + [det("washer", 0.9, x1=900)], spec)
    assert status == Status.FAIL and [(f.code, f.class_name) for f in findings] == [("EXTRA", "washer")]


def test_wrong_class_fails(spec):
    # a washer read as a nut: 3 nuts + 1 washer is wrong in both directions and is not a screw confusion
    status, findings = verify_product(
        parts(["screw_long", "screw_short", "nut", "nut", "nut", "washer"]), spec)
    assert status == Status.FAIL
    assert {(f.code, f.class_name) for f in findings} == {("EXTRA", "nut"), ("MISSING", "washer")}


def test_uncertain_detection_is_review_never_pass(spec):
    dets = parts()
    dets[2] = det("nut", 0.45, x1=130)
    status, findings = verify_product(dets, spec)
    assert status == Status.REVIEW and codes(findings) == ["UNCERTAIN_PART"]
    # every part just under accept must also not PASS
    assert verify_product(parts(conf=0.59), spec)[0] != Status.PASS


# ---- long vs short screw competition ----------------------------------------
def test_same_object_read_as_long_and_short_is_review(spec):
    dets = parts() + [det("screw_short", 0.45, x1=10)]  # box identical to the screw_long box
    status, findings = verify_product(dets, spec)
    assert status == Status.REVIEW and codes(findings) == ["COMPETING_CLASSES"]
    f = findings[0]
    assert f.class_name == "screw_long" and "Check it by eye" in f.message
    assert f.evidence["kept"]["class"] == "screw_long" and f.evidence["dropped"]["class"] == "screw_short"


def test_both_screws_read_as_long_is_review_with_clear_text(spec):
    status, findings = verify_product(
        parts(["screw_long", "screw_long", "nut", "nut", "washer", "washer"]), spec)
    assert status == Status.REVIEW and codes(findings) == ["CLASS_SPLIT"]
    f = findings[0]
    assert "screw_long 2 (expected 1)" in f.message and "screw_short 0 (expected 1)" in f.message
    assert "by eye" in f.message and f.evidence == {"expected": {"screw_long": 1, "screw_short": 1},
                                                  "found": {"screw_long": 2, "screw_short": 0},
                                                  "low_confidence": {"screw_long": 0, "screw_short": 0}}


def test_short_screw_possibly_hiding_as_low_confidence_long_is_review(spec):
    # seen on real normals: 1 confident long + 1 low-confidence long, no short. Enough screw candidates,
    # unclear which is which: REVIEW, not "MISSING screw_short" FAIL.
    dets = parts(["screw_long", "nut", "nut", "washer", "washer"]) + [det("screw_long", 0.45, x1=700)]
    status, findings = verify_product(dets, spec)
    assert status == Status.REVIEW and codes(findings) == ["CLASS_SPLIT", "UNCERTAIN_EXTRA"]
    split = next(f for f in findings if f.code == "CLASS_SPLIT")
    assert "1 confident, 1 low-confidence" in split.message and "screw_long 1 + 1 low-confidence" in split.message


def test_missing_screw_with_no_covering_candidate_is_still_a_fail(spec):
    status, findings = verify_product(parts(["screw_long", "nut", "nut", "washer", "washer"]), spec)
    assert status == Status.FAIL and [(f.code, f.class_name) for f in findings] == [("MISSING", "screw_short")]
    # a low-confidence NUT does not cover a missing screw: the group is screws only
    status, findings = verify_product(
        parts(["screw_long", "nut", "nut", "washer", "washer"]) + [det("nut", 0.45, x1=700)], spec)
    assert status == Status.FAIL and ("MISSING", "screw_short") in [(f.code, f.class_name) for f in findings]


def test_both_screws_read_as_short_is_review(spec):
    status, findings = verify_product(
        parts(["screw_short", "screw_short", "nut", "nut", "washer", "washer"]), spec)
    assert status == Status.REVIEW and codes(findings) == ["CLASS_SPLIT"]


def test_wrong_total_of_screws_is_still_a_fail(spec):
    # three confident screws: not a classification question, there is a screw too many
    status, findings = verify_product(
        parts(["screw_long", "screw_long", "screw_short", "nut", "nut", "washer", "washer"]), spec)
    assert status == Status.FAIL and codes(findings) == ["EXTRA"]
    # no screw at all
    status, findings = verify_product(parts(["nut", "nut", "washer", "washer"]), spec)
    assert status == Status.FAIL and codes(findings) == ["MISSING", "MISSING"]


def test_class_split_plus_real_missing_part_still_fails(spec):
    status, findings = verify_product(
        parts(["screw_long", "screw_long", "nut", "washer", "washer"]), spec)
    assert status == Status.FAIL
    assert {(f.code, f.class_name) for f in findings} == {("CLASS_SPLIT", "screw_long/screw_short"),
                                                          ("MISSING", "nut")}


def test_noise_below_review_conf_does_not_trigger_competition(spec):
    dets = parts() + [det("screw_short", 0.1, x1=10)]
    assert verify_product(dets, spec) == (Status.PASS, [])


def test_spec_without_competing_classes_keeps_original_behaviour():
    plain = ProductSpec(product="t", version="1", thresholds=Thresholds(accept_conf=0.6, review_conf=0.3),
                        required_counts={"screw_long": 1, "screw_short": 1})
    status, findings = verify_counts(parts(["screw_long", "screw_long"]), plain)
    assert status == Status.FAIL and codes(findings) == ["EXTRA", "MISSING"]


def test_result_is_deterministic_and_order_independent(spec):
    dets = parts(["screw_long", "screw_long", "nut", "washer", "washer"]) + [det("nut", 0.4, x1=800)]
    base = verify_product(dets, spec)
    assert verify_product(list(reversed(dets)), spec)[0] == base[0]
    assert sorted((f.code, f.class_name) for f in verify_product(list(reversed(dets)), spec)[1]) == \
           sorted((f.code, f.class_name) for f in base[1])
    assert verify_product(dets, spec) == base


# ---- spec validation --------------------------------------------------------
def test_competing_group_must_use_known_classes():
    t = Thresholds(accept_conf=0.6, review_conf=0.3)
    with pytest.raises(ValueError):
        ProductSpec(product="t", version="1", thresholds=t, required_counts={"a": 1},
                    competing_classes=[["a", "b"]])
    with pytest.raises(ValueError):
        ProductSpec(product="t", version="1", thresholds=t, required_counts={"a": 1, "b": 1},
                    competing_classes=[["a"]])
