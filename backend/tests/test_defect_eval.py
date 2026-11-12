import numpy as np
import pytest
from PIL import Image

from app.datasets.loco import Sample
from app.datasets.testsplit import assign_split, stratum_key, write_frozen
from app.evaluation.verdicts import summarize_verdicts


def make_strata():
    st = {f"test/good/{i:03d}.png": "good" for i in range(20)}
    st.update({f"test/logical_anomalies/m{i}.png": "missing_nut" for i in range(10)})
    st.update({f"test/logical_anomalies/r{i}.png": "rare" for i in range(2)})
    st["test/logical_anomalies/single.png"] = "single"
    return st


def count(st, a, key, part):
    return sum(1 for r, k in st.items() if k == key and a[r] == part)


def test_assign_split_properties():
    st = make_strata()
    a = assign_split(st, seed=1)
    assert a == assign_split(st, seed=1)  # deterministic
    assert set(a) == set(st) and set(a.values()) <= {"dev", "final"}
    assert (count(st, a, "good", "dev"), count(st, a, "good", "final")) == (6, 14)
    assert (count(st, a, "missing_nut", "dev"), count(st, a, "missing_nut", "final")) == (3, 7)
    assert (count(st, a, "rare", "dev"), count(st, a, "rare", "final")) == (1, 1)  # n=2: both parts
    assert (count(st, a, "single", "dev"), count(st, a, "single", "final")) == (0, 1)  # n=1: sealed


def test_assign_split_seed_changes_membership():
    st = make_strata()
    base = assign_split(st, seed=1)
    assert any(assign_split(st, seed=s) != base for s in range(2, 8))


def test_stratum_key(tmp_path):
    mask = tmp_path / "m.png"
    Image.fromarray(np.array([[0, 255], [254, 0]], dtype=np.uint8)).save(mask)
    s = Sample("c", "test", "logical_anomalies", tmp_path / "i.png", "test/logical_anomalies/000.png", (mask,))
    assert stratum_key(s, {255: "b_def", 254: "a_def"}) == "a_def+b_def"
    assert stratum_key(s, {255: "b_def"}) == "UNKNOWN_254+b_def"
    normal = Sample("c", "test", "good", tmp_path / "g.png", "test/good/000.png")
    assert stratum_key(normal, {}) == "good"


def test_write_frozen_creates_once_and_refuses_changes(tmp_path):
    rec = {"assignment": {"a": "dev"}, "strata": {"a": "good"}}
    p = tmp_path / "s" / "split.json"
    assert write_frozen(p, rec) == "created"
    assert write_frozen(p, rec) == "unchanged"
    with pytest.raises(ValueError, match="frozen"):
        write_frozen(p, {"assignment": {"a": "final"}, "strata": {"a": "good"}})


def test_summarize_verdicts():
    rows = [{"group": "normal", "stratum": "good", "verdict": "PASS"},
            {"group": "normal", "stratum": "good", "verdict": "FAIL"},
            {"group": "logical", "stratum": "x", "verdict": "REVIEW"},
            {"group": "logical", "stratum": "x", "verdict": "PASS"},
            {"group": "logical", "stratum": "y", "verdict": "FAIL"}]
    per, grp = summarize_verdicts(rows)
    assert per["good"] == {"PASS": 1, "REVIEW": 0, "FAIL": 1, "n": 2, "group": "normal"}
    assert per["x"]["REVIEW"] == 1 and per["x"]["n"] == 2 and per["x"]["group"] == "logical"
    assert grp["logical"] == {"PASS": 1, "REVIEW": 1, "FAIL": 1, "n": 3}


def test_summarize_verdicts_rejects_unknown():
    with pytest.raises(ValueError):
        summarize_verdicts([{"group": "normal", "stratum": "good", "verdict": "MAYBE"}])