from collections import defaultdict

VERDICTS = ("PASS", "REVIEW", "FAIL")


def _blank():
    return {"PASS": 0, "REVIEW": 0, "FAIL": 0, "n": 0}


def summarize_verdicts(rows):
    """rows: dicts with 'group' (normal/logical/structural), 'stratum' and 'verdict'.
    Returns (per-stratum tallies, per-group tallies)."""
    per, grp = defaultdict(_blank), defaultdict(_blank)
    group_of_stratum = {}
    for r in rows:
        v = r["verdict"]
        if v not in VERDICTS:
            raise ValueError(f"unknown verdict {v!r}")
        for tally in (per[r["stratum"]], grp[r["group"]]):
            tally[v] += 1
            tally["n"] += 1
        group_of_stratum[r["stratum"]] = r["group"]
    per_out = {k: {**t, "group": group_of_stratum[k]} for k, t in per.items()}
    return per_out, dict(grp)