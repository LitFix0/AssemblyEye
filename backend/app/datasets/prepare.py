import csv
import shutil
from pathlib import Path

from .loco import index_category
from .selection import select_for_annotation

MANIFEST_FIELDS = ["file", "role", "source"]


def flat_name(sample) -> str:
    return f"{sample.split}_{Path(sample.rel_path).name}"  # e.g. train_000.png


def read_manifest(path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prepare_batch(root, category, classes, n_train_pool, n_holdout, out_dir, seed=42) -> list[dict]:
    chosen = select_for_annotation(index_category(root, category), n_train_pool, n_holdout, seed)
    rows = [{"file": flat_name(s), "role": role, "source": s.rel_path} for s, role in chosen]

    cat_out = Path(out_dir) / category
    manifest_path = cat_out / "manifest.csv"
    new = {r["file"]: r for r in rows}
    stale = [f for f, r in {r["file"]: r for r in read_manifest(manifest_path)}.items()
             if new.get(f) != r]
    if stale:
        raise ValueError(f"existing manifest has {len(stale)} entries not in the new selection "
                         "(shrinking the batch, or a changed seed/dataset?); refusing to overwrite")

    (cat_out / "images").mkdir(parents=True, exist_ok=True)
    (cat_out / "labels_json").mkdir(exist_ok=True)
    sources = {flat_name(s): s.image_path for s, _ in chosen}
    for r in rows:
        dst = cat_out / "images" / r["file"]
        if not dst.exists():
            shutil.copy2(sources[r["file"]], dst)
    (cat_out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows