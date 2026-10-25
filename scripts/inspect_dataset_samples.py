"""ST-00: verify mask encoding and build contact sheets for visual review."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("data/raw/mvtec_loco")
CATEGORIES = ["screw_bag", "splicing_connectors"]
KINDS = ["logical_anomalies", "structural_anomalies"]
OUT = Path("reports/samples")
OUT.mkdir(parents=True, exist_ok=True)


def check_masks(cat: str) -> None:
    base = ROOT / cat
    for kind in KINDS:
        test_stems = {p.stem for p in (base / "test" / kind).glob("*.png")}
        mask_dirs = {d.name: d for d in (base / "ground_truth" / kind).iterdir() if d.is_dir()}
        print(f"\n{cat}/{kind}")
        print("  test images:", len(test_stems), "| mask dirs:", len(mask_dirs))
        print("  test images without mask dir:", sorted(test_stems - set(mask_dirs))[:5])
        print("  mask dirs without test image:", sorted(set(mask_dirs) - test_stems)[:5])
        per_image = Counter(len(list(d.glob("*.png"))) for d in mask_dirs.values())
        print("  masks-per-image histogram:", dict(per_image))
        values = Counter()
        for d in mask_dirs.values():
            for m in d.glob("*.png"):
                values.update(np.unique(np.array(Image.open(m))).tolist())
        print("  pixel values (masks containing each):", dict(sorted(values.items())))


def thumb(img: Image.Image, width: int = 400) -> Image.Image:
    return img.resize((width, int(img.height * width / img.width)))


def overlay(img: Image.Image, mask_paths) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    region = np.zeros(arr.shape[:2], dtype=bool)
    for p in mask_paths:
        region |= np.array(Image.open(p)) > 0
    arr[region] = (0.5 * arr[region] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    return Image.fromarray(arr)


def save_row(images, path: Path) -> None:
    ims = [thumb(i) for i in images]
    sheet = Image.new("RGB", (sum(i.width for i in ims), max(i.height for i in ims)))
    x = 0
    for im in ims:
        sheet.paste(im, (x, 0))
        x += im.width
    sheet.save(path, quality=85)


def pick(paths, n: int = 4):
    paths = sorted(paths)
    step = max(len(paths) // n, 1)
    return paths[::step][:n]


def make_sheets(cat: str) -> None:
    base = ROOT / cat
    for split, sub in [("train_good", "train/good"), ("test_good", "test/good")]:
        imgs = [Image.open(p).convert("RGB") for p in pick((base / sub).glob("*.png"))]
        save_row(imgs, OUT / f"{cat}_{split}.jpg")
    for kind in KINDS:
        raws, ovs = [], []
        for p in pick((base / "test" / kind).glob("*.png")):
            img = Image.open(p).convert("RGB")
            masks = list((base / "ground_truth" / kind / p.stem).glob("*.png"))
            raws.append(img)
            ovs.append(overlay(img, masks))
        save_row(raws, OUT / f"{cat}_{kind}_raw.jpg")
        save_row(ovs, OUT / f"{cat}_{kind}_overlay.jpg")


if __name__ == "__main__":
    for c in CATEGORIES:
        check_masks(c)
        make_sheets(c)
    print("\nContact sheets written to", OUT)