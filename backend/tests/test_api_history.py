"""ST-07: history through the HTTP API (fake detector, temporary database)."""
import dataclasses
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.settings import load_settings
from app.main import create_app
from app.schemas.inspection import BBox, Detection

REPO = Path(__file__).resolve().parents[2]


def box(cls, conf, x1, y1=20, w=40, h=40):
    return Detection(class_name=cls, confidence=conf, bbox=BBox(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h))


GOOD = [box("screw_long", 0.9, 10), box("screw_short", 0.9, 60), box("nut", 0.9, 110), box("nut", 0.9, 160),
        box("washer", 0.9, 210), box("washer", 0.9, 260)]
NO_WASHERS = [d for d in GOOD if d.class_name != "washer"]


class Fake:
    name = "fake:test"
    class_names = []

    def __init__(self):
        self.dets = GOOD

    def detect(self, bgr):
        return list(self.dets)


def png() -> bytes:
    return cv2.imencode(".png", np.full((300, 400, 3), 110, dtype=np.uint8))[1].tobytes()


@pytest.fixture
def env(tmp_path):
    settings = dataclasses.replace(load_settings(), data_dir=tmp_path / "images", db_path=tmp_path / "h.db")
    fake = Fake()

    def client():  # a fresh app over the same files = a server restart
        return TestClient(create_app(settings, lambda e: fake), raise_server_exceptions=False)
    return client, fake, settings


def inspect(client, name="a.png"):
    r = client.post("/inspect", data={"product": "screw_bag"}, files={"file": (name, png(), "image/png")})
    assert r.status_code == 200
    return r.json()


def test_every_inspection_appears_in_history_newest_first(env):
    client, fake, _ = env
    c = client()
    first = inspect(c, "first.png")
    fake.dets = NO_WASHERS
    second = inspect(c, "second.png")
    body = c.get("/inspections").json()
    assert body["total"] == 2 and body["limit"] == 50 and body["offset"] == 0
    assert [i["inspection_id"] for i in body["items"]] == [second["inspection_id"], first["inspection_id"]]
    row = body["items"][0]
    assert row["status"] == "FAIL" and row["finding_codes"] == ["MISSING"] and row["image_name"] == "second.png"
    assert row["product"] == "screw_bag" and row["detector"] == "fake:test"


def test_filters_by_status_code_and_product(env):
    client, fake, _ = env
    c = client()
    inspect(c)
    fake.dets = NO_WASHERS
    failed = inspect(c)
    assert [i["inspection_id"] for i in c.get("/inspections", params={"status": "FAIL"}).json()["items"]] == [failed["inspection_id"]]
    assert c.get("/inspections", params={"code": "MISSING"}).json()["total"] == 1
    assert c.get("/inspections", params={"status": "PASS"}).json()["total"] == 1
    assert c.get("/inspections", params={"product": "splicing_connectors"}).json()["total"] == 0


def test_date_filters_use_the_requested_bounds(env):
    client, _, _ = env
    c = client()
    inspect(c)
    now = datetime.now(timezone.utc)
    hour = timedelta(hours=1)
    assert c.get("/inspections", params={"since": (now - hour).isoformat()}).json()["total"] == 1
    assert c.get("/inspections", params={"since": (now + hour).isoformat()}).json()["total"] == 0
    assert c.get("/inspections", params={"until": (now - hour).isoformat()}).json()["total"] == 0


def test_paging(env):
    client, _, _ = env
    c = client()
    for i in range(3):
        inspect(c, f"{i}.png")
    page = c.get("/inspections", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 3 and len(page["items"]) == 1


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"status": "MAYBE"},
                                    {"since": "yesterday-ish"}])
def test_bad_query_parameters_are_422_not_500(env, params):
    client, _, _ = env
    r = client().get("/inspections", params=params)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_history_and_details_survive_a_restart(env):
    client, _, _ = env
    done = inspect(client())
    restarted = client()
    assert restarted.get("/inspections").json()["total"] == 1
    again = restarted.get(f"/inspections/{done['inspection_id']}").json()
    assert again["result"] == done["result"]
    assert restarted.get(done["evidence"]["annotated_url"]).status_code == 200


def test_delete_removes_record_and_images(env):
    client, _, settings = env
    c = client()
    done = inspect(c)
    iid = done["inspection_id"]
    assert (settings.data_dir / iid).is_dir()
    assert c.delete(f"/inspections/{iid}").status_code == 204
    assert c.get(f"/inspections/{iid}").status_code == 404
    assert c.get(f"/inspections/{iid}/annotated").status_code == 404
    assert not (settings.data_dir / iid).exists()
    assert c.get("/inspections").json()["total"] == 0
    assert c.delete(f"/inspections/{iid}").status_code == 404  # second delete: nothing left


@pytest.mark.parametrize("iid", ["nope", "../etc", "0" * 32])
def test_delete_of_unknown_or_hostile_ids_is_404(env, iid):
    client, _, _ = env
    assert client().delete(f"/inspections/{iid}").status_code == 404


def test_a_database_failure_leaves_no_images_and_no_row(env):
    client, _, settings = env
    c = client()

    def boom(_):
        raise RuntimeError("disk is full")
    c.app.state.store.add = boom
    r = c.post("/inspect", data={"product": "screw_bag"}, files={"file": ("a.png", png(), "image/png")})
    assert r.status_code == 500 and "disk is full" not in r.text
    assert list(settings.data_dir.iterdir()) == []


# ---- scripts ---------------------------------------------------------------------
def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_import_script_loads_old_result_json_folders_and_is_idempotent(env, capsys):
    client, _, settings = env
    c = client()
    done = inspect(c)
    iid = done["inspection_id"]
    # simulate an ST-05 folder: a result.json on disk, no database row
    (settings.data_dir / iid / "result.json").write_text(c.get(f"/inspections/{iid}").text, encoding="utf-8")
    c.app.state.store.delete(iid)
    junk = settings.data_dir / ("a" * 32)
    junk.mkdir()
    (junk / "result.json").write_text("{not json", encoding="utf-8")
    mod = script("import_inspections")
    args = ["--db", str(settings.db_path), "--data-dir", str(settings.data_dir)]
    assert mod.main(args) == 0
    out = capsys.readouterr()
    assert "imported 1, already present 0, unreadable 1" in out.out and "aaaaaaaa" in out.err
    assert client().get(f"/inspections/{iid}").json()["result"] == done["result"]
    assert mod.main(args) == 0
    assert "imported 0, already present 1" in capsys.readouterr().out


def test_purge_script_dry_run_changes_nothing_and_real_run_deletes_rows_and_folders(env, capsys):
    client, _, settings = env
    c = client()
    iid = inspect(c)["inspection_id"]
    mod = script("purge_inspections")
    args = ["--db", str(settings.db_path), "--data-dir", str(settings.data_dir)]
    assert mod.main(["--days", "0", "--dry-run", *args]) == 0
    assert "dry run" in capsys.readouterr().out
    assert c.get("/inspections").json()["total"] == 1 and (settings.data_dir / iid).is_dir()
    assert mod.main(["--days", "30", *args]) == 0           # too recent to qualify
    assert "0 inspection(s)" in capsys.readouterr().out
    assert mod.main(["--days", "0", *args]) == 0
    assert "deleted 1 record" in capsys.readouterr().out
    assert c.get("/inspections").json()["total"] == 0 and not (settings.data_dir / iid).exists()
    assert mod.main(["--days", "-1", *args]) == 2