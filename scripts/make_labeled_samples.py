"""ST-00: one labeled example per defect type, high-contrast mask overlay."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("data/raw/mvtec_loco")
OUT = Path("reports/samples_labeled")
OUT.mkdir(parents=True, exist_ok=True)
CATEGORIES = ["screw_bag", "splicing_connectors"]
KINDS = ["logical_anomalies", "structural_anomalies"]
W, COLS, LABEL_H = 400, 4, 32


def defect_names(base: Path) -> dict:
    cfg = json.loads((base / "defects_config.json").read_text())
    return {d["pixel_value"]: d["defect_name"] for d in cfg}


def values_of(mask_paths) -> list:
    vals = set()
    for m in mask_paths:
        vals |= set(np.unique(np.array(Image.open(m))).tolist())
    vals.discard(0)
    return sorted(vals)


def thick_boundary(region: np.ndarray, n: int = 3) -> np.ndarray:
    inner = (region & np.roll(region, 1, 0) & np.roll(region, -1, 0)
             & np.roll(region, 1, 1) & np.roll(region, -1, 1))
    b = region & ~inner
    for _ in range(n):
        b = (b | np.roll(b, 1, 0) | np.roll(b, -1, 0)
             | np.roll(b, 1, 1) | np.roll(b, -1, 1))
    return b


def render(img: Image.Image, mask_paths) -> Image.Image:
    arr = np.array(img)
    region = np.zeros(arr.shape[:2], dtype=bool)
    for m in mask_paths:
        region |= np.array(Image.open(m)) > 0
    out = arr.copy()
    out[region] = (0.65 * arr[region] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
    out[thick_boundary(region)] = (0, 255, 0)
    return Image.fromarray(out)


def make_sheet(cat: str, kind: str) -> None:
    base = ROOT / cat
    names = defect_names(base)
    examples = {}  # pixel value -> (image path, mask paths), first one seen
    for p in sorted((base / "test" / kind).glob("*.png")):
        masks = sorted((base / "ground_truth" / kind / p.stem).glob("*.png"))
        for v in values_of(masks):
            examples.setdefault(v, (p, masks))

    cells = []
    for v, (p, masks) in sorted(examples.items(), reverse=True):
        cell = render(Image.open(p).convert("RGB"), masks)
        cell = cell.resize((W, int(cell.height * W / cell.width)))
        d = ImageDraw.Draw(cell)
        d.rectangle([0, 0, W, LABEL_H], fill=(0, 0, 0))
        d.text((4, 2), f"{kind[:4]}/{p.name}", fill="white")
        d.text((4, 16), f"{v}: {names.get(v, '?')}"[:60], fill="yellow")
        cells.append(cell)

    if not cells:
        return
    cell_h = max(c.height for c in cells)
    rows = (len(cells) + COLS - 1) // COLS
    sheet = Image.new("RGB", (W * COLS, cell_h * rows))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % COLS) * W, (i // COLS) * cell_h))
    sheet.save(OUT / f"{cat}_{kind}.jpg", quality=85)
    print(f"{cat}/{kind}: {len(cells)} defect types -> {cat}_{kind}.jpg")


if __name__ == "__main__":
    for c in CATEGORIES:
        for k in KINDS:
            make_sheet(c, k)