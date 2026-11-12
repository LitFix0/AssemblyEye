"""One-time, frozen, stratified split of the official test images into DEV and FINAL.

DEV is for threshold tuning and error analysis. FINAL is sealed: nothing may be tuned on it, and it is
evaluated once, at the end, with the finished system.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .loco import Sample, defect_values

GROUPS = {"good": "normal", "logical_anomalies": "logical", "structural_anomalies": "structural"}


def stratum_key(sample: Sample, names: dict[int, str]) -> str:
    """'good' for normals; otherwise the sorted '+'-joined defect names found in the image's masks."""
    if sample.is_normal:
        return "good"
    vals = defect_values(sample)
    return "+".join(sorted(names.get(v, f"UNKNOWN_{v}") for v in vals)) or "UNKNOWN"


def assign_split(strata: dict[str, str], seed: int = 42, dev_fraction: float = 0.3) -> dict[str, str]:
    """strata: {rel_path: stratum}. Deterministic. Within each stratum, images are ranked by a seeded
    hash and the first k go to dev. k = 0 for strata with fewer than 2 images (they stay sealed),
    otherwise round(n * dev_fraction) clipped to [1, n - 1]."""
    groups = defaultdict(list)
    for rel, key in strata.items():
        groups[key].append(rel)
    out = {}
    for key, rels in sorted(groups.items()):
        ordered = sorted(rels, key=lambda r: hashlib.sha1(f"{seed}:{r}".encode()).hexdigest())
        n = len(ordered)
        k = 0 if n < 2 else min(n - 1, max(1, round(n * dev_fraction)))
        for i, rel in enumerate(ordered):
            out[rel] = "dev" if i < k else "final"
    return out


def build_split(samples, names, seed: int = 42, dev_fraction: float = 0.3) -> dict:
    test = [s for s in samples if s.split == "test"]
    strata = {s.rel_path: stratum_key(s, names) for s in test}
    groups = {s.rel_path: GROUPS[s.label] for s in test}
    assignment = assign_split(strata, seed, dev_fraction)
    counts = defaultdict(lambda: {"dev": 0, "final": 0})
    for rel, key in strata.items():
        counts[key][assignment[rel]] += 1
    return {"seed": seed, "dev_fraction": dev_fraction, "n_test": len(test),
            "counts": dict(sorted(counts.items())), "assignment": dict(sorted(assignment.items())),
            "strata": dict(sorted(strata.items())), "groups": dict(sorted(groups.items()))}


def write_frozen(path, record: dict) -> str:
    """Create the split file once. If it exists, the regenerated split must be identical."""
    p = Path(path)
    if p.is_file():
        existing = json.loads(p.read_text(encoding="utf-8"))
        if existing.get("assignment") == record["assignment"] and existing.get("strata") == record["strata"]:
            return "unchanged"
        raise ValueError(f"{p} already exists and differs from the regenerated split; it is frozen")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return "created"


def load_split(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))