import pytest

from app.evaluation.screw_length import diagonal, fit_length_threshold, relabel_by_length
from app.schemas.inspection import BBox, Detection


def det(cls, w, h, conf=0.9):
    return Detection(class_name=cls, confidence=conf, bbox=BBox(x1=0, y1=0, x2=w, y2=h))


def test_diagonal():
    assert diagonal(det("screw", 300, 400)) == pytest.approx(500)


def test_fit_separable_uses_gap_midpoint():
    fit = fit_length_threshold([440, 480, 520], [330, 370, 420])
    assert fit["separable"] and fit["threshold"] == pytest.approx(430)
    assert fit["margin_px"] == pytest.approx(20) and fit["wrong"] == 0 and fit["total"] == 6


def test_fit_overlap_uses_best_accuracy():
    fit = fit_length_threshold([344, 450, 480], [330, 370, 445])
    assert not fit["separable"] and fit["threshold"] == 450 and fit["wrong"] == 1


def test_fit_requires_both_classes():
    with pytest.raises(ValueError):
        fit_length_threshold([400], [])


def test_relabel_by_length_changes_only_merged_screws():
    dets = [det("screw", 300, 400), det("screw", 100, 100), det("nut", 50, 50)]
    out = relabel_by_length(dets, 450)
    assert [d.class_name for d in out] == ["screw_long", "screw_short", "nut"]
    assert dets[0].class_name == "screw"  # input untouched