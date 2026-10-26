import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.datasets.loco import (DatasetLayoutError, defect_values, index_category,  # noqa: E402
                               load_defect_names, summarize)

CATEGORIES = ["screw_bag", "splicing_connectors"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Index MVTec LOCO AD with the project adapter.")
    ap.add_argument("--root", default="data/raw/mvtec_loco")
    args = ap.parse_args(argv)
    try:
        for cat in CATEGORIES:
            samples = index_category(args.root, cat)
            names = load_defect_names(Path(args.root) / cat)
            print(cat)
            for key, n in summarize(samples).items():
                print(f"  {key}: {n}")
            print(f"  mask files: {sum(len(s.mask_paths) for s in samples)}")
            per_defect: Counter = Counter()
            for s in samples:
                for v in defect_values(s):
                    per_defect[names.get(v, f"UNKNOWN_{v}")] += 1
            for name, n in sorted(per_defect.items()):
                print(f"    {name}: {n} image(s)")
    except (DatasetLayoutError, OSError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())