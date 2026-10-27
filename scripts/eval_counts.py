import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.image_io import load_image  # noqa: E402
from app.datasets.loco import DatasetLayoutError, index_category  # noqa: E402
from app.datasets.prepare import read_manifest  # noqa: E402
from app.datasets.selection import select_normal_dev  # noqa: E402
from app.evaluation.counts import (counts_at, exact_screw_bag, exact_splicing,  # noqa: E402
                                   suppress_cross_class, summarize)
from app.schemas.spec import load_spec  # noqa: E402
from app.services.detection.yolo_detector import YoloDetector  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Count-level evaluation on the normal-only dev set.")
    ap.add_argument("--category", required=True, choices=["screw_bag", "splicing_connectors"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--root", default="data/raw/mvtec_loco")
    ap.add_argument("--annotation-dir", default="data/annotation")
    ap.add_argument("--spec", default="configs/products/screw_bag.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--worst", type=int, default=5)
    args = ap.parse_args(argv)
    try:
        dev = select_normal_dev(index_category(args.root, args.category), args.n, args.seed)
        used = {r["source"] for r in read_manifest(Path(args.annotation_dir) / args.category / "manifest.csv")}
        overlap = [s.rel_path for s in dev if s.rel_path in used]
        if overlap:
            raise ValueError(f"{len(overlap)} dev image(s) are also in the annotation manifest: {overlap[:3]}")
        if args.category == "screw_bag":
            required = load_spec(args.spec).required_counts

            def check(c):
                return exact_screw_bag(c, required)
        else:
            check = exact_splicing
        detector = YoloDetector(args.weights, args.imgsz,
                                min_conf=min(0.10, min(args.thresholds)), device=args.device)
        per_image = [detector.detect(load_image(s.image_path).bgr) for s in dev]
    except (DatasetLayoutError, ValueError, OSError, ImportError, KeyError, yaml.YAMLError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    raw = summarize(per_image, check, args.thresholds)
    sup = summarize([suppress_cross_class(d) for d in per_image], check, args.thresholds)
    print(f"\n{args.category}: {len(dev)} normal dev images (train/good, never annotated, never trained on)")
    print(f"detector: {detector.name}\n")
    print(f"{'thr':>5}  {'exact (raw)':>14}  {'exact (suppressed)':>20}  {'imgs w/ overlap':>16}")
    for a, b in zip(raw, sup):
        ra = f"{a['exact']}/{a['images']} ({a['exact_rate']:.0%})"
        rb = f"{b['exact']}/{b['images']} ({b['exact_rate']:.0%})"
        print(f"{a['threshold']:>5.2f}  {ra:>14}  {rb:>20}  {a['images_with_cross_class_overlap']:>16}")

    ref = 0.5 if 0.5 in args.thresholds else args.thresholds[len(args.thresholds) // 2]
    failures = {s.image_path.name: dict(sorted(counts_at(d, ref).items()))
                for s, d in zip(dev, per_image) if not check(counts_at(d, ref))}
    print(f"\nfailing at threshold {ref} (raw): {len(failures)}; first {args.worst}:")
    for name, c in list(failures.items())[:args.worst]:
        print(f"  {name}: {c}")

        out = Path("reports") / f"eval_counts_{args.category}_{Path(args.weights).stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"category": args.category, "detector": detector.name, "n": len(dev),
                               "seed": args.seed, "images": [s.rel_path for s in dev],
                               "raw": raw, "suppressed": sup, "failures_at": ref,
                               "failures": failures}, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())