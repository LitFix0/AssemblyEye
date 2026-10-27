import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.schemas.spec import load_spec  # noqa: E402
from app.datasets.yolo_export import export_yolo  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export labelme annotations to a YOLO dataset.")
    ap.add_argument("--category", required=True)
    ap.add_argument("--annotation-dir", default="data/annotation")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--classes", default="configs/datasets/loco_classes.yaml")
    ap.add_argument("--spec", default=None, help="product spec for the composition check (screw_bag)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--strict", action="store_true", help="fail if any manifest image is unlabelled")
    args = ap.parse_args(argv)
    try:
        classes = yaml.safe_load(Path(args.classes).read_text(encoding="utf-8"))[args.category]
        spec_path = args.spec or ("configs/products/screw_bag.yaml" if args.category == "screw_bag" else None)
        spec_counts = load_spec(spec_path).required_counts if spec_path else None
        out_dir = Path(args.out) / f"yolo_{args.category}"
        report = export_yolo(Path(args.annotation_dir) / args.category, classes, out_dir,
                             args.seed, args.val_fraction, spec_counts, args.strict)
        Path("reports").mkdir(exist_ok=True)
        Path(f"reports/st02_export_{args.category}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    except (ValueError, OSError, KeyError, yaml.YAMLError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    im = report["images"]
    print(f"{args.category}: train {im['train']} / val {im['val']} / holdout {im['holdout']} -> {out_dir}")
    print("boxes per class:", report["boxes_per_class"])
    print("annotation shapes converted to boxes:", report["source_shapes"])
    if report["unlabelled"]:
        print(f"unlabelled (skipped): {len(report['unlabelled'])}")

    if report["unreviewed"]:
        print(f"UNREVIEWED drafts (skipped): {len(report['unreviewed'])}")
    if report["composition_issues"]:
        print(f"COMPOSITION ISSUES in {len(report['composition_issues'])} image(s):")
        for name, msgs in report["composition_issues"].items():
            print(f"  {name}: {'; '.join(msgs)}")
    else:
        print("composition check: every labelled image matches the expected assembly")
    return 0


if __name__ == "__main__":
    sys.exit(main())