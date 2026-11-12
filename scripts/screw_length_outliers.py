import sys
from math import hypot
from pathlib import Path

import yaml
from PIL import Image


def main() -> int:
    ds = Path("data/processed/yolo_screw_bag")
    try:
        names = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))["names"]
        rows = []  # (diagonal px, class, split, image stem)
        for lbl in sorted((ds / "labels").rglob("*.txt")):
            split = lbl.parent.name
            with Image.open(ds / "images" / split / f"{lbl.stem}.png") as im:
                w, h = im.size
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                c, _, _, bw, bh = line.split()
                if names[int(c)] in ("screw_long", "screw_short"):
                    rows.append((hypot(float(bw) * w, float(bh) * h), names[int(c)], split, lbl.stem))
    except (OSError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    longs = [r[0] for r in rows if r[1] == "screw_long"]
    shorts = [r[0] for r in rows if r[1] == "screw_short"]
    best_t, best_ok = 0.0, -1
    for t in sorted(longs + shorts):
        ok = sum(v >= t for v in longs) + sum(v < t for v in shorts)
        if ok > best_ok:
            best_t, best_ok = t, ok
    print(f"threshold {best_t:.0f} px: {best_ok}/{len(rows)} correct\n")

    wrong = [r for r in rows if (r[1] == "screw_long") != (r[0] >= best_t)]
    print("on the wrong side of the threshold:")
    for d, cls, split, stem in sorted(wrong):
        print(f"  {stem}.png ({split})  labelled {cls}  diagonal {d:.0f} px")
    print("\nnearest to the threshold (either side):")
    for d, cls, split, stem in sorted(rows, key=lambda r: abs(r[0] - best_t))[:8]:
        print(f"  {stem}.png ({split})  labelled {cls}  diagonal {d:.0f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())