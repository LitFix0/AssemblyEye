import argparse
import sys
from math import hypot
from pathlib import Path

import yaml
from PIL import Image


def stats(vals):
    s = sorted(vals)
    return f"n={len(s):>3}  min={s[0]:.0f}  p10={s[len(s)//10]:.0f}  median={s[len(s)//2]:.0f}  p90={s[(9*len(s))//10]:.0f}  max={s[-1]:.0f}"


def best_threshold(long_v, short_v):
    best = (0.0, -1)
    for t in sorted(long_v + short_v):
        correct = sum(v >= t for v in long_v) + sum(v < t for v in short_v)
        if correct > best[1]:
            best = (t, correct)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Do long and short screws separate by labelled box size?")
    ap.add_argument("--dataset", default="data/processed/yolo_screw_bag")
    args = ap.parse_args()
    ds = Path(args.dataset)
    try:
        names = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))["names"]
        diag = {"screw_long": [], "screw_short": []}
        side = {"screw_long": [], "screw_short": []}
        for lbl in sorted((ds / "labels").rglob("*.txt")):
            with Image.open(ds / "images" / lbl.parent.name / f"{lbl.stem}.png") as im:
                w, h = im.size
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                c, _, _, bw, bh = line.split()
                name = names[int(c)]
                if name in diag:
                    bw_px, bh_px = float(bw) * w, float(bh) * h
                    diag[name].append(hypot(bw_px, bh_px))
                    side[name].append(max(bw_px, bh_px))
    except (OSError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    for title, d in (("box diagonal (px)", diag), ("longest box side (px)", side)):
        print(f"\n{title}")
        for cls in ("screw_long", "screw_short"):
            print(f"  {cls:<12} {stats(d[cls])}")
        t, ok = best_threshold(d["screw_long"], d["screw_short"])
        total = len(d["screw_long"]) + len(d["screw_short"])
        print(f"  best single threshold: {t:.0f} px -> {ok}/{total} screws correct ({total - ok} wrong)")
        print(f"  largest short = {max(d['screw_short']):.0f}, smallest long = {min(d['screw_long']):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())