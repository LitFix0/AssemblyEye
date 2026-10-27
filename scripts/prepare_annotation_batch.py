import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.datasets.loco import DatasetLayoutError  # noqa: E402
from app.datasets.prepare import prepare_batch  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select normal images and set up the annotation workspace.")
    ap.add_argument("--category", required=True)
    ap.add_argument("--train-pool", type=int, required=True)
    ap.add_argument("--holdout", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default="data/raw/mvtec_loco")
    ap.add_argument("--out", default="data/annotation")
    ap.add_argument("--classes", default="configs/datasets/loco_classes.yaml")
    ap.add_argument("--reserve-dev", type=int, default=60)
    args = ap.parse_args(argv)
    try:
        cfg = yaml.safe_load(Path(args.classes).read_text(encoding="utf-8"))
        if args.category not in cfg:
            raise ValueError(f"no classes defined for '{args.category}'")
        rows = prepare_batch(args.root, args.category, cfg[args.category],
                             args.train_pool, args.holdout, args.out, args.seed)
    except (DatasetLayoutError, ValueError, OSError, yaml.YAMLError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    n_hold = sum(r["role"] == "holdout" for r in rows)
    print(f"{args.category}: {len(rows)} images ({len(rows) - n_hold} train_pool, {n_hold} holdout)"
          f" -> {Path(args.out) / args.category}")
    return 0


if __name__ == "__main__":
    sys.exit(main())