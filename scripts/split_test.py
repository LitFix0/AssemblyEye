import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.datasets.loco import DatasetLayoutError, index_category, load_defect_names  # noqa: E402
from app.datasets.testsplit import build_split, write_frozen  # noqa: E402

CATEGORIES = ["screw_bag", "splicing_connectors"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create the frozen DEV/FINAL split of the test images.")
    ap.add_argument("--root", default="data/raw/mvtec_loco")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-fraction", type=float, default=0.3)
    args = ap.parse_args(argv)
    try:
        for cat in CATEGORIES:
            samples = index_category(args.root, cat)
            names = load_defect_names(Path(args.root) / cat)
            rec = build_split(samples, names, args.seed, args.dev_fraction)
            status = write_frozen(Path(args.out_dir) / f"test_split_{cat}.json", rec)
            asg, groups = rec["assignment"], rec["groups"]
            print(f"\n{cat}: split {status} ({rec['n_test']} test images)")
            for g in ("normal", "logical", "structural"):
                dev = sum(1 for r in asg if groups[r] == g and asg[r] == "dev")
                fin = sum(1 for r in asg if groups[r] == g and asg[r] == "final")
                print(f"  {g:<11} dev {dev:>3}   final {fin:>3}")
            print("  defect type (dev / final):")
            for key, c in rec["counts"].items():
                if key != "good":
                    print(f"    {key:<46}{c['dev']:>3} / {c['final']:>3}")
    except (DatasetLayoutError, ValueError, OSError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())