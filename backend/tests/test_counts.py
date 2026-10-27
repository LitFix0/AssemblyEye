from collections import Counter

import pytest

from app.evaluation.counts import (counts_at, cross_class_pairs, exact_screw_bag, exact_splicing,
                                   iou, summarize, suppress_cross_class)
from app.schemas.inspection import BBox, Detection

REQ = {"screw_long": 1, "screw_short": 1, "nut": 2, "washer": 2}


def det(cls, conf, x1=0, y1=0, x2=100, y2=100):
    return Detection(class_name=cls, confidence=conf, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))


def parts(classes):
    return [det(c, .9, x1=i * 200, x2=i * 200 + 100) for i, c in enumerate(classes)]


def test_iou():
    a, b = BBox(x1=0, y1=0, x2=10, y2=10), BBox(x1=5, y1=0, x2=15, y2=10)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, b) == pytest.approx(50 / 150)
    assert iou(a, BBox(x1=20, y1=20, x2=30, y2=30)) == 0.0


def test_counts_at_and_exact_screw_bag():
    dets = [det("screw_long", .9), det("screw_short", .8), det("nut", .7), det("nut", .6),
            det("washer", .9), det("washer", .5), det("washer", .2)]
    assert counts_at(dets, 0.5) == Counter({"screw_long": 1, "screw_short": 1, "nut": 2, "washer": 2})
    assert exact_screw_bag(counts_at(dets, 0.5), REQ)
    assert not exact_screw_bag(counts_at(dets, 0.2), REQ)  # extra washer
    assert not exact_screw_bag(counts_at(dets, 0.65), REQ)  # missing nut and washer
    assert not exact_screw_bag(Counter({**REQ, "bolt": 1}), REQ)  # unexpected class


def test_exact_splicing():
    assert exact_splicing(Counter({"connector_3": 2, "cable": 1}))
    assert not exact_splicing(Counter({"connector_3": 2}))
    assert not exact_splicing(Counter({"connector_3": 2, "cable": 2}))
    assert not exact_splicing(Counter({"connector_3": 1, "connector_5": 1, "cable": 1}))
    assert not exact_splicing(Counter({"connector_3": 3, "cable": 1}))
    assert not exact_splicing(Counter({"connector_3": 2, "cable": 1, "nut": 1}))


def test_cross_class_pairs():
    dets = [det("nut", .8), det("washer", .6, x1=2, x2=102), det("nut", .8, x1=500, x2=600),
            det("washer", .2)]
    assert cross_class_pairs(dets, 0.5) == 1
    assert cross_class_pairs(dets, 0.9) == 0
    assert cross_class_pairs(dets, 0.1) == 2


def test_suppress_cross_class_keeps_higher_confidence_and_same_class_boxes():
    dets = [det("nut", .5), det("washer", .6, x1=2, x2=102),
            det("washer", .3, x1=300, x2=400), det("washer", .25, x1=302, x2=402)]
    out = suppress_cross_class(dets)
    assert [(d.class_name, d.confidence) for d in out] == [("washer", .6), ("washer", .3), ("washer", .25)]


def test_summarize():
    good = parts(["screw_long", "screw_short", "nut", "nut", "washer", "washer"])
    bad = parts(["screw_long", "screw_short", "nut", "nut", "washer"])
    rows = summarize([good, bad], lambda c: exact_screw_bag(c, REQ), [0.5, 0.95])
    assert rows[0]["exact"] == 1 and rows[0]["images"] == 2 and rows[0]["exact_rate"] == 0.5
    assert rows[1]["exact"] == 0  # nothing survives 0.95
    assert rows[0]["cross_class_pairs"] == 0