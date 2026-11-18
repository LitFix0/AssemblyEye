"""ST-05: the HTTP layer with a fake detector (no weights, no GPU, no Ultralytics needed)."""
import dataclasses

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.image_io import MAX_BYTES
from app.core.settings import load_settings
from app.main import create_app
from app.schemas.inspection import BBox, Detection


def box(cls, conf, x1, y1=20, w=40, h=40):
    return Detection(class_name=cls, confidence=conf, bbox=BBox(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h))


NORMAL_SCREW_BAG = [box("screw_long", 0.9, 10), box("screw_short", 0.9, 60), box("nut", 0.9, 110),
                    box("nut", 0.9, 160), box("washer", 0.9, 210), box("washer", 0.9, 260)]


class FakeDetector:
    name = "fake:test-detector"
    class_names = []

    def __init__(self, dets):
        self.dets = dets
        self.calls = 0

    def detect(self, bgr):
        self.calls += 1
        return list(self.dets)


def png(w=400, h=300, value=110) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), value, dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def make_client(tmp_path):
    def _make(dets=NORMAL_SCREW_BAG, factory=None):
        settings = dataclasses.replace(load_settings(), data_dir=tmp_path / "inspections")
        fake = FakeDetector(dets)
        app = create_app(settings, factory or (lambda entry: fake))
        return TestClient(app, raise_server_exceptions=False), fake, settings
    return _make


def post(client, data, name="img.png", product="screw_bag"):
    return client.post("/inspect", data={"product": product}, files={"file": (name, data, "image/png")})


# ---- health and products ---------------------------------------------------------
def test_health_lists_products_and_reports_missing_weights(make_client):
    client, _, _ = make_client()
    body = client.get("/health").json()
    assert {p["product"] for p in body["products"]} == {"screw_bag", "splicing_connectors"}
    # weights are not committed, so on a fresh checkout the service must say "degraded", not crash
    assert body["status"] in ("ok", "degraded")
    assert all(p["detector_loaded"] is False for p in body["products"])
    assert client.get("/products").status_code == 200


# ---- successful inspections ------------------------------------------------------
def test_normal_screw_bag_passes_and_evidence_is_retrievable(make_client):
    client, fake, _ = make_client()
    r = post(client, png(), "my part.png")
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["status"] == "PASS" and body["result"]["findings"] == []
    assert body["result"]["product"] == "screw_bag" and body["result"]["detector"] == "fake:test-detector"
    assert body["result"]["image"]["name"] == "my part.png"
    assert set(body["timings_ms"]) == {"load", "detect", "total"}
    for kind in ("original", "annotated"):
        img = client.get(body["evidence"][f"{kind}_url"])
        assert img.status_code == 200 and img.headers["content-type"].startswith("image/")
    stored = client.get(f"/inspections/{body['inspection_id']}")
    assert stored.status_code == 200 and stored.json()["result"] == body["result"]
    assert fake.calls == 1


def test_missing_part_fails_with_a_reason(make_client):
    client, _, _ = make_client(dets=[d for d in NORMAL_SCREW_BAG if d.class_name != "washer"])
    body = post(client, png()).json()
    assert body["result"]["status"] == "FAIL"
    assert any(f["code"] == "MISSING" and f["class_name"] == "washer" for f in body["result"]["findings"])


def test_uncertain_detection_is_review_never_pass(make_client):
    dets = [box("screw_long", 0.45, 10)] + NORMAL_SCREW_BAG[1:]
    client, _, _ = make_client(dets=dets)
    assert post(client, png()).json()["result"]["status"] == "REVIEW"


def test_splicing_runs_with_pixels_and_unreadable_colour_is_review(make_client):
    dets = [Detection(class_name="connector_2", confidence=0.9, bbox=BBox(x1=10, y1=60, x2=80, y2=140)),
            Detection(class_name="connector_2", confidence=0.9, bbox=BBox(x1=320, y1=60, x2=390, y2=140)),
            Detection(class_name="cable", confidence=0.5, bbox=BBox(x1=78, y1=90, x2=322, y2=110))]
    client, _, _ = make_client(dets=dets)
    body = post(client, png(400, 200), product="splicing_connectors").json()
    assert body["result"]["status"] == "REVIEW"
    assert [f["code"] for f in body["result"]["findings"]] == ["CABLE_COLOR_UNKNOWN"]


# ---- rejected requests ------------------------------------------------------------
def test_unknown_product_is_404(make_client):
    client, fake, settings = make_client()
    r = post(client, png(), product="toaster")
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_product"
    assert fake.calls == 0


@pytest.mark.parametrize("name,data,code", [
    ("notes.txt", b"hello", "invalid_image"),            # wrong type
    ("noext", png(), "invalid_image"),                     # no extension
    ("bad.png", b"this is not an image", "invalid_image"),  # right extension, not decodable
    ("empty.png", b"", "invalid_image"),
    ("tiny.png", png(10, 10), "invalid_image"),            # below the minimum side
])
def test_invalid_uploads_are_422_and_leave_no_evidence(make_client, name, data, code):
    client, fake, settings = make_client()
    r = post(client, data, name)
    assert r.status_code == 422 and r.json()["error"]["code"] == code
    assert fake.calls == 0 and list(settings.data_dir.iterdir()) == []


def test_oversized_upload_is_413(make_client):
    client, fake, _ = make_client()
    r = post(client, b"\0" * (MAX_BYTES + 1))
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"
    assert fake.calls == 0


def test_missing_fields_are_422(make_client):
    client, _, _ = make_client()
    assert client.post("/inspect", data={"product": "screw_bag"}).status_code == 422
    assert client.post("/inspect", files={"file": ("a.png", png(), "image/png")}).status_code == 422


def test_unloadable_detector_is_503_not_a_crash(make_client):
    def broken(entry):
        raise FileNotFoundError("weights not found: models/x.pt")
    client, _, settings = make_client(factory=broken)
    r = post(client, png())
    assert r.status_code == 503 and r.json()["error"]["code"] == "detector_unavailable"
    assert list(settings.data_dir.iterdir()) == []


def test_unexpected_error_is_500_without_leaking_details(make_client):
    class Exploding(FakeDetector):
        def detect(self, bgr):
            raise RuntimeError("secret internal detail")
    client, _, settings = make_client(factory=lambda entry: Exploding([]))
    r = post(client, png())
    assert r.status_code == 500 and "secret" not in r.text
    assert r.json()["error"]["code"] == "internal_error"
    assert list(settings.data_dir.iterdir()) == []


# ---- evidence lookup safety -------------------------------------------------------
@pytest.mark.parametrize("path", ["/inspections/nope", "/inspections/" + "0" * 32,
                                  "/inspections/..%2f..%2fetc/original", "/inspections/" + "0" * 32 + "/other"])
def test_unknown_or_hostile_inspection_ids_are_404(make_client, path):
    client, _, _ = make_client()
    assert client.get(path).status_code == 404


def test_client_filename_never_becomes_a_path(make_client):
    client, _, settings = make_client()
    body = post(client, png(), "../../evil name.png").json()
    assert body["result"]["image"]["name"] == "evil name.png"
    stored = sorted(p.name for p in (settings.data_dir / body["inspection_id"]).iterdir())
    assert stored == ["annotated.png", "original.png", "result.json"]  # fixed names, nothing from the client
    assert not any(settings.data_dir.parent.glob("evil*"))


def test_screw_bag_applies_cross_class_suppression_like_the_dev_evaluation(make_client):
    # one object read as both screws: with suppression on (serving.yaml) the engine never sees the pair
    dets = [box("screw_long", 0.9, 10), box("screw_short", 0.8, 10)] + NORMAL_SCREW_BAG[2:]
    client, _, _ = make_client(dets=dets)
    codes = [f["code"] for f in post(client, png()).json()["result"]["findings"]]
    assert "COMPETING_CLASSES" not in codes and "MISSING" in codes


def test_folder_that_is_not_an_inspection_id_is_404_even_if_it_exists(make_client):
    client, _, settings = make_client()
    (settings.data_dir / "not-an-id").mkdir()
    assert client.get("/inspections/not-an-id").status_code == 404