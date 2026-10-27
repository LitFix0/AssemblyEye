import json

from app.datasets.labelme_io import is_unreviewed, parse_labelme
from app.datasets.prelabel import build_labelme, cap_to_expected
from app.schemas.inspection import BBox, Detection

REQ = {"screw_long": 1, "screw_short": 1, "nut": 2, "washer": 2}


def det(cls, conf, x1=10, y1=10, x2=50, y2=50):
    return Detection(class_name=cls, confidence=conf, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))


def test_cap_screw_bag_keeps_top_k_per_class_and_drops_unknown():
    dets = [det("nut", .9), det("nut", .8), det("nut", .7), det("washer", .6), det("bolt", .95),
            det("screw_long", .5), det("screw_long", .4)]
    out = cap_to_expected(dets, "screw_bag", REQ)
    assert sorted((d.class_name, d.confidence) for d in out) == sorted(
        [("nut", .9), ("nut", .8), ("washer", .6), ("screw_long", .5)])


def test_cap_splicing_keeps_two_connectors_and_one_cable():
    dets = [det("connector_3", .9), det("connector_5", .8), det("connector_2", .3),
            det("cable", .4), det("cable", .2)]
    out = cap_to_expected(dets, "splicing_connectors")
    assert [d.class_name for d in out] == ["connector_3", "connector_5", "cable"]


def test_build_labelme_roundtrips_and_is_flagged(tmp_path):
    template = {"version": "x", "flags": {}, "imagePath": "..\\images\\old.png", "imageData": None,
                "imageHeight": 80, "imageWidth": 100,
                "shapes": [{"label": "a", "points": [[1, 1], [5, 5], [9, 1]], "group_id": None,
                            "description": "", "shape_type": "polygon", "flags": {}, "mask": None}]}
    data = build_labelme(template, "new.png", 100, 80, [det("nut", .8)])
    assert data["imagePath"] == "..\\images\\new.png"
    assert data["flags"] == {"needs_review": True} and data["imageData"] is None
    assert data["shapes"][0]["shape_type"] == "rectangle" and data["shapes"][0]["mask"] is None
    p = tmp_path / "new.json"
    p.write_text(json.dumps(data))
    boxes = parse_labelme(p, ["nut"], image_size=(100, 80))
    assert (boxes[0].label, boxes[0].x1, boxes[0].y2) == ("nut", 10, 50)
    assert is_unreviewed(p)
    assert template["shapes"][0]["shape_type"] == "polygon"  # template untouched