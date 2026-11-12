from app.datasets.variants import merge_classes, remap_label_lines

NAMES = {0: "screw_long", 1: "screw_short", 2: "nut", 3: "washer"}


def test_merge_classes_order_and_mapping():
    names, mapping = merge_classes(NAMES, {"screw_long": "screw", "screw_short": "screw"})
    assert names == ["screw", "nut", "washer"]
    assert mapping == {0: 0, 1: 0, 2: 1, 3: 2}


def test_remap_label_lines_keeps_geometry():
    text = "0 0.1 0.2 0.3 0.4\n1 0.5 0.5 0.1 0.1\n\n3 0.9 0.9 0.2 0.2\n"
    out = remap_label_lines(text, {0: 0, 1: 0, 2: 1, 3: 2})
    assert out == "0 0.1 0.2 0.3 0.4\n0 0.5 0.5 0.1 0.1\n2 0.9 0.9 0.2 0.2\n"