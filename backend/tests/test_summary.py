from pathlib import Path

import pytest

from app.schemas.inspection import BBox, Detection
from app.schemas.spec import load_spec
from app.services.verification.summary import component_table

PRODUCTS = Path(__file__).resolve().parents[2] / "configs" / "products"


def det(cls, conf, x1=0):
    return Detection(class_name=cls, confidence=conf, bbox=BBox(x1=x1, y1=0, x2=x1 + 10, y2=10))


def rows(table):
    return {r.class_name: (r.expected, r.confident, r.uncertain) for r in table}


def test_screw_bag_table_counts_bands_per_class():
    spec = load_spec(PRODUCTS / "screw_bag.yaml")
    dets = [det("screw_long", 0.9), det("nut", 0.9), det("nut", 0.45), det("washer", 0.1)]  # 0.1 is ignored
    assert rows(component_table(dets, spec)) == {
        "screw_long": (1, 1, 0), "screw_short": (1, 0, 0), "nut": (2, 1, 1), "washer": (2, 0, 0)}


def test_splicing_table_groups_connector_types_and_uses_cable_thresholds():
    spec = load_spec(PRODUCTS / "splicing_connectors.yaml")
    dets = [det("connector_2", 0.9), det("connector_3", 0.9), det("cable", 0.45)]  # cable accept is 0.40
    assert rows(component_table(dets, spec)) == {"connector": (2, 2, 0), "cable": (1, 1, 0)}


def test_empty_detections_give_zero_rows_not_an_empty_table():
    spec = load_spec(PRODUCTS / "screw_bag.yaml")
    assert all(r.confident == 0 and r.uncertain == 0 for r in component_table([], spec))
    assert len(component_table([], spec)) == 4