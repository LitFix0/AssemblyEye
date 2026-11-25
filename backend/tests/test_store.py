"""ST-07: the SQLite history store, without HTTP."""
from datetime import datetime, timedelta, timezone

import pytest

from app.db.store import InspectionStore
from app.schemas.api import Evidence, InspectionResponse
from app.schemas.inspection import Finding, ImageInfo, InspectionResult, Status

T0 = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)


def make(i: int, product="screw_bag", status=Status.PASS, codes=(), when=None, name=None) -> InspectionResponse:
    findings = [Finding(code=c, status=Status.FAIL if status == Status.FAIL else Status.REVIEW, class_name="x",
                        message=f"{c} happened") for c in codes]
    iid = f"{i:032x}"
    return InspectionResponse(
        inspection_id=iid, created_at=when or (T0 + timedelta(minutes=i)),
        result=InspectionResult(product=product, spec_version="0.2", status=status,
                                image=ImageInfo(name=name or f"img{i}.png", width=100, height=80),
                                detections=[], findings=findings, detector="fake", components=[]),
        evidence=Evidence(original_url=f"/inspections/{iid}/original", annotated_url=f"/inspections/{iid}/annotated"),
        timings_ms={"load": 1.0, "detect": 2.0, "total": 3.5})


@pytest.fixture
def store(tmp_path):
    return InspectionStore(tmp_path / "db" / "t.db")  # parent folder is created on demand


def ids(result):
    return [r.inspection_id for r in result.items]


def test_round_trip_keeps_the_full_result(store):
    original = make(1, status=Status.FAIL, codes=["MISSING"])
    store.add(original)
    assert store.get(original.inspection_id) == original
    assert store.get("f" * 32) is None


def test_history_survives_a_restart(tmp_path):
    path = tmp_path / "t.db"
    InspectionStore(path).add(make(1))
    reopened = InspectionStore(path)  # a new process would do exactly this
    assert reopened.search().total == 1 and reopened.get(f"{1:032x}") is not None


def test_search_is_newest_first_with_a_total_that_ignores_paging(store):
    for i in range(1, 6):
        store.add(make(i))
    page = store.search(limit=2, offset=0)
    assert ids(page) == [f"{5:032x}", f"{4:032x}"] and page.total == 5
    assert ids(store.search(limit=2, offset=4)) == [f"{1:032x}"]


def test_filters_combine(store):
    store.add(make(1, "screw_bag", Status.PASS))
    store.add(make(2, "screw_bag", Status.FAIL, ["MISSING"]))
    store.add(make(3, "splicing_connectors", Status.FAIL, ["CABLE_TOO_SHORT", "MISSING"]))
    store.add(make(4, "splicing_connectors", Status.REVIEW, ["UNCERTAIN_PART"]))
    assert ids(store.search(product="splicing_connectors")) == [f"{4:032x}", f"{3:032x}"]
    assert ids(store.search(status="FAIL")) == [f"{3:032x}", f"{2:032x}"]
    assert ids(store.search(code="MISSING")) == [f"{3:032x}", f"{2:032x}"]
    assert ids(store.search(product="splicing_connectors", status="FAIL", code="MISSING")) == [f"{3:032x}"]
    assert store.search(code="NOPE").total == 0


def test_summary_lists_each_finding_code_once_sorted(store):
    store.add(make(1, status=Status.FAIL, codes=["MISSING", "EXTRA", "MISSING"]))
    store.add(make(2))
    by_id = {r.inspection_id: r for r in store.search().items}
    assert by_id[f"{1:032x}"].finding_codes == ["EXTRA", "MISSING"]
    assert by_id[f"{2:032x}"].finding_codes == []


def test_since_is_inclusive_and_until_is_exclusive(store):
    for i in range(3):
        store.add(make(i, when=T0 + timedelta(days=i)))  # Jan 10, 11, 12 at noon UTC
    assert len(store.search(since=T0 + timedelta(days=1)).items) == 2
    assert len(store.search(until=T0 + timedelta(days=1)).items) == 1  # the one exactly at the bound is out
    assert len(store.search(since=T0 + timedelta(days=1), until=T0 + timedelta(days=2)).items) == 1


def test_timezone_aware_bounds_are_compared_in_utc(store):
    store.add(make(1, when=T0))  # 12:00 UTC
    ist = timezone(timedelta(hours=5, minutes=30))
    assert store.search(since=datetime(2026, 1, 10, 17, 30, tzinfo=ist)).total == 1   # = 12:00 UTC, inclusive
    assert store.search(since=datetime(2026, 1, 10, 17, 31, tzinfo=ist)).total == 0


def test_delete_removes_the_row_and_its_findings(store):
    store.add(make(1, status=Status.FAIL, codes=["MISSING"]))
    assert store.delete(f"{1:032x}") is True
    assert store.delete(f"{1:032x}") is False
    assert store.search(code="MISSING").total == 0  # findings went with it (cascade)


def test_ids_older_than(store):
    store.add(make(1, when=T0))
    store.add(make(2, when=T0 + timedelta(days=30)))
    assert store.ids_older_than(T0 + timedelta(days=1)) == [f"{1:032x}"]


def test_duplicate_id_is_rejected_and_leaves_one_row(store):
    store.add(make(1))
    with pytest.raises(Exception):
        store.add(make(1))
    assert store.search().total == 1


@pytest.mark.filterwarnings("ignore")
def test_a_failed_findings_insert_rolls_back_the_inspection_row(store):
    bad = make(1, status=Status.FAIL, codes=["MISSING"])
    bad.result.findings[0].code = None  # not validated on assignment; the NOT NULL constraint must stop it
    with pytest.raises(Exception):
        store.add(bad)
    assert store.search().total == 0  # the inspection row written just before it was rolled back too