"""Export the annotation workspace (manifest + labelme JSON) to a YOLO detection dataset."""
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from .labelme_io import AnnotationError, is_unreviewed, parse_labelme, to_yolo_line
from .prepare import read_manifest
from .selection import LeakageError

ROLE_SOURCE_PREFIX = {"train_pool": "train/good/", "holdout": "validation/good/"}


def assert_manifest_clean(rows) -> None:
    for r in rows:
        prefix = ROLE_SOURCE_PREFIX.get(r["role"])
        if prefix is None or not r["source"].startswith(prefix):
            raise LeakageError(f"{r['file']}: role {r['role']!r} is not allowed for source {r['source']!r}")


def split_train_pool(files, seed: int = 42, val_fraction: float = 0.2):
    """Deterministic split by seeded hash rank. Returns (train, val), each sorted."""
    ordered = sorted(files, key=lambda f: hashlib.sha1(f"{seed}:{f}".encode()).hexdigest())
    n = len(ordered)
    n_val = 0 if n < 2 else min(n - 1, max(1, round(n * val_fraction)))
    return sorted(ordered[n_val:]), sorted(ordered[:n_val])


def composition_issues(category: str, counts, spec_counts=None) -> list[str]:
    """Compare one labelled NORMAL image with the expected assembly (a data-quality report)."""
    issues = []
    if category == "screw_bag" and spec_counts:
        for cls, n in spec_counts.items():
            if counts.get(cls, 0) != n:
                issues.append(f"{cls}: expected {n}, labelled {counts.get(cls, 0)}")
        for cls in sorted(set(counts) - set(spec_counts)):
            issues.append(f"{cls}: not in spec, labelled {counts[cls]}")
    elif category == "splicing_connectors":
        connectors = {c: n for c, n in counts.items() if c.startswith("connector_")}
        if sum(connectors.values()) != 2 or len(connectors) != 1:
            issues.append(f"expected 2 connectors of one type, labelled {connectors or 'none'}")
        if counts.get("cable", 0) != 1:
            issues.append(f"expected 1 cable, labelled {counts.get('cable', 0)}")
    return issues


def read_yolo_labels(path, width: int, height: int):
    """Return [(class_index, x1, y1, x2, y2)] in pixel coordinates."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        cx, bw = float(cx) * width, float(bw) * width
        cy, bh = float(cy) * height, float(bh) * height
        out.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out


def export_yolo(category_dir, classes, out_dir, seed=42, val_fraction=0.2,
                spec_counts=None, strict=False) -> dict:
    cat = Path(category_dir)
    rows = read_manifest(cat / "manifest.csv")
    if not rows:
        raise ValueError(f"no manifest found in {cat}")
    assert_manifest_clean(rows)

    parsed, unlabelled, unreviewed, errors = {}, [], [], []
    for r in rows:
        name = r["file"]
        jpath = cat / "labels_json" / f"{Path(name).stem}.json"
        if not jpath.is_file():
            unlabelled.append(name)
            continue
        if is_unreviewed(jpath):
            unreviewed.append(name)
            continue
        with Image.open(cat / "images" / name) as im:
            size = im.size
        try:
            parsed[name] = parse_labelme(jpath, classes, size)
        except AnnotationError as e:
            errors.append(str(e))
    if errors:
        raise AnnotationError(f"{len(errors)} annotation problem(s):\n  " + "\n  ".join(errors))
    if strict and unlabelled:
        raise ValueError(f"{len(unlabelled)} unlabelled image(s): {', '.join(unlabelled[:5])} ...")
    if not parsed:
        raise ValueError("no labelled images found")

    role = {r["file"]: r["role"] for r in rows}
    pool = [n for n in parsed if role[n] == "train_pool"]
    train, val = split_train_pool(pool, seed, val_fraction)
    if not train or not val:
        raise ValueError("need at least 2 labelled train_pool images to create train and val splits")
    holdout = sorted(n for n in parsed if role[n] == "holdout")
    assignment = {**{n: "train" for n in train}, **{n: "val" for n in val},
                  **{n: "holdout" for n in holdout}}
    assert len(assignment) == len(parsed)

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        if not (out / "data.yaml").is_file():
            raise ValueError(f"refusing to clear {out}: it does not look like a previous export")
        shutil.rmtree(out)
    for split in set(assignment.values()):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    per_class: Counter = Counter()
    shape_counts: Counter = Counter()
    issues: dict = {}
    for name, split in sorted(assignment.items()):
        boxes = parsed[name]
        src = cat / "images" / name
        with Image.open(src) as im:
            w, h = im.size
        shutil.copy2(src, out / "images" / split / name)
        lines = [to_yolo_line(b, classes.index(b.label), w, h) for b in boxes]
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        counts = Counter(b.label for b in boxes)
        per_class.update(counts)
        shape_counts.update(b.source_shape for b in boxes)
        found = composition_issues(cat.name, counts, spec_counts)
        if found:
            issues[name] = found

    data_yaml = {"train": "images/train", "val": "images/val"}
    if holdout:
        data_yaml["test"] = "images/holdout"
    data_yaml["names"] = dict(enumerate(classes))
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    report = {
        "category": cat.name, "seed": seed, "val_fraction": val_fraction, "classes": list(classes),
        "images": {"train": len(train), "val": len(val), "holdout": len(holdout)},
        "boxes_per_class": {c: per_class.get(c, 0) for c in classes},
        "source_shapes": dict(shape_counts),
        "unlabelled": sorted(unlabelled), "composition_issues": issues,
        "unreviewed": sorted(unreviewed),
    }
    (out / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report