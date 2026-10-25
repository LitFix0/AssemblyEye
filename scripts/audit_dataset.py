import json, sys
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path("data/raw/mvtec_loco")
CATEGORIES = ["screw_bag", "splicing_connectors"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def tree(path: Path, depth=0, max_depth=3):
    if depth > max_depth:
        return
    children = sorted(path.iterdir())
    dirs = [c for c in children if c.is_dir()]
    files = [c for c in children if c.is_file()]
    if files:
        ext = Counter(f.suffix.lower() for f in files)
        print("  " * depth + f"{path.name}/  [{dict(ext)}]")
    else:
        print("  " * depth + f"{path.name}/")
    for d in dirs[:6]:  # cap fan-out (mask dirs can be numerous)
        tree(d, depth + 1, max_depth)
    if len(dirs) > 6:
        print("  " * (depth + 1) + f"... {len(dirs) - 6} more subdirs")


def count_images(path: Path):
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMG_EXT)


def audit(cat: str):
    base = ROOT / cat
    print("=" * 70)
    print(cat, "->", base)
    if not base.exists():
        print("MISSING"); return
    tree(base)

    print("\nImage counts:")
    for split in ["train", "validation", "test", "ground_truth"]:
        for sub in sorted((base / split).glob("*")) if (base / split).exists() else []:
            if sub.is_dir():
                print(f"  {split}/{sub.name}: {count_images(sub)}")

    sizes = Counter()
    modes = Counter()
    for p in list((base / "train").rglob("*"))[:50]:
        if p.suffix.lower() in IMG_EXT:
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
    print("\nTrain image sizes (sample):", dict(sizes))
    print("Train image modes (sample):", dict(modes))

    print("\nNon-image files (annotation/metadata candidates):")
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix.lower() not in IMG_EXT:
            print("  ", p.relative_to(base))
            if p.suffix.lower() == ".json":
                try:
                    data = json.loads(p.read_text())
                    print("     preview:", json.dumps(data, indent=1)[:600])
                except Exception as e:
                    print("     (could not parse:", e, ")")

    gt = base / "ground_truth"
    if gt.exists():
        mask_dirs = [d for d in gt.rglob("*") if d.is_dir() and any(d.glob("*.png"))]
        if mask_dirs:
            sample = next(mask_dirs[0].glob("*.png"))
            with Image.open(sample) as im:
                print("\nSample mask:", sample.relative_to(base), im.size, im.mode)


if __name__ == "__main__":
    for c in CATEGORIES:
        audit(c)