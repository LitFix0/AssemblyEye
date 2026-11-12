import argparse
import json
import sys
from datetime import datetime, timezone
from math import hypot
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.evaluation.screw_length import fit_length_threshold  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit the long/short screw length threshold on TRAIN + VAL labels only.")
    ap.add_argument("--dataset", default="data/processed/yolo_screw_bag", help="the TWO-class dataset")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--out", default="reports/screw_length_threshold.json")
    args = ap.parse_args(argv)
    ds = Path(args.dataset)
    try:
        names = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))["names"]
        longs, shorts = [], []
        for split in args.splits:
            for lbl in sorted((ds / "labels" / split).glob("*.txt")):
                with Image.open(ds / "images" / split / f"{lbl.stem}.png") as im:
                    w, h = im.size
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    c, _, _, bw, bh = line.split()
                    d = hypot(float(bw) * w, float(bh) * h)
                    if names[int(c)] == "screw_long":
                        longs.append(d)
                    elif names[int(c)] == "screw_short":
                        shorts.append(d)
        fit = fit_length_threshold(longs, shorts)
    except (OSError, KeyError, ValueError, yaml.YAMLError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    record = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "dataset": str(ds), "splits": args.splits, "n_long": len(longs), "n_short": len(shorts),
              "smallest_long": round(min(longs), 1), "largest_short": round(max(shorts), 1),
              **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in fit.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())