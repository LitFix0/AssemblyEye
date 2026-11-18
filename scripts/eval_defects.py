import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.image_io import load_image  # noqa: E402
from app.datasets.loco import DatasetLayoutError, index_category  # noqa: E402
from app.datasets.testsplit import load_split  # noqa: E402
from app.evaluation.counts import suppress_cross_class  # noqa: E402
from app.evaluation.verdicts import summarize_verdicts  # noqa: E402
from app.schemas.spec import Thresholds, load_spec  # noqa: E402
from app.services.detection.yolo_detector import YoloDetector  # noqa: E402
from app.services.verification.engine import verify_product  # noqa: E402


def fmt(label, t):
    flagged = (t["REVIEW"] + t["FAIL"]) / t["n"]
    return f"{label:<46}{t['n']:>4}{t['PASS']:>6}{t['REVIEW']:>8}{t['FAIL']:>6}{flagged:>9.0%}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end PASS/REVIEW/FAIL evaluation on the DEV test split.")
    ap.add_argument("--category", default="screw_bag", choices=["screw_bag", "splicing_connectors"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--split", default="dev", choices=["dev"],
                    help="the FINAL split is sealed and has its own once-only script")
    ap.add_argument("--accept", type=float, default=None,
                    help="override the spec's GLOBAL accept_conf (per-class thresholds are not changed)")
    ap.add_argument("--review", type=float, default=None,
                    help="override the spec's GLOBAL review_conf (per-class thresholds are not changed)")
    ap.add_argument("--spec", default=None, help="default: configs/products/<category>.yaml")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--root", default="data/raw/mvtec_loco")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-suppress", action="store_true",
                    help="screw_bag only: skip the external cross-class suppression. The verification "
                         "engine then handles one-object-two-classes itself and reports it as REVIEW")
    args = ap.parse_args(argv)
    # External suppression only exists for screw_bag; splicing handles competing classes in the engine.
    suppress = args.category == "screw_bag" and not args.no_suppress
    try:
        spec = load_spec(args.spec or f"configs/products/{args.category}.yaml")
        if args.accept is not None or args.review is not None:
            spec = spec.model_copy(update={"thresholds": Thresholds(
                accept_conf=args.accept if args.accept is not None else spec.thresholds.accept_conf,
                review_conf=args.review if args.review is not None else spec.thresholds.review_conf)})
        split = load_split(args.split_file or f"data/splits/test_split_{args.category}.json")
        dev = [r for r, part in split["assignment"].items() if part == args.split]
        samples = {s.rel_path: s for s in index_category(args.root, args.category)}
        missing = [r for r in dev if r not in samples]
        if missing:
            raise ValueError(f"{len(missing)} split image(s) not found in the dataset: {missing[:3]}")
        detector = YoloDetector(args.weights, args.imgsz,
                                min_conf=min(0.10, spec.min_review_conf), device=args.device)
        if args.category == "screw_bag" and "screw" in detector.class_names:
            raise ValueError("this is the merged one-class screw detector; use the two-class (A) weights")
        if (args.category == "splicing_connectors") != (spec.splicing is not None):
            raise ValueError(f"spec {spec.product!r} does not match category {args.category!r}")
        rows = []
        for rel in dev:
            img = load_image(samples[rel].image_path)
            dets = detector.detect(img.bgr)
            if suppress:
                dets = suppress_cross_class(dets)
            status, findings = verify_product(dets, spec, img.bgr)
            rows.append({"image": rel, "group": split["groups"][rel], "stratum": split["strata"][rel],
                         "verdict": status.value, "findings": [f"{f.code}:{f.class_name}" for f in findings],
                         "messages": [f.message for f in findings]})
    except (DatasetLayoutError, ValueError, OSError, ImportError, KeyError, yaml.YAMLError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    per, grp = summarize_verdicts(rows)
    t = spec.thresholds
    print(f"\n{args.category}  {args.split.upper()} split: {len(rows)} images")
    print(f"detector: {detector.name}")
    print(f"spec: {spec.product} v{spec.version}")
    print(f"thresholds: accept {t.accept_conf:g}, review {t.review_conf:g}"
          + "".join(f"; {c}: accept {ct.accept_conf:g}, review {ct.review_conf:g}"
                    for c, ct in spec.class_thresholds.items()))
    print(f"cross-class suppression before the rules: {'on' if suppress else 'off'}\n")
    print(f"{'group / defect type':<46}{'n':>4}{'PASS':>6}{'REVIEW':>8}{'FAIL':>6}{'flagged':>9}")
    for group in ("normal", "logical", "structural"):
        if group not in grp:
            continue
        print(fmt(f"== {group}", grp[group]))
        if group != "normal":
            for key, tally in sorted(per.items()):
                if tally["group"] == group:
                    print(fmt(f"   {key}", tally))
    flagged_normals = Counter(c for r in rows if r["group"] == "normal" and r["verdict"] != "PASS"
                              for c in set(r["findings"]))
    if flagged_normals:
        print("\nwhy normal images were flagged (finding code:class -> images):")
        for code, n in flagged_normals.most_common():
            print(f"   {code:<44}{n:>4}")
    out = Path("reports") / (f"eval_defects_{args.category}_{Path(args.weights).stem}"
                             f"_a{t.accept_conf:g}_r{t.review_conf:g}"
                             f"{'_nosup' if args.category == 'screw_bag' and args.no_suppress else ''}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"category": args.category, "detector": detector.name, "split": args.split,
                               "suppress": suppress, "spec_version": spec.version,
                               "accept": t.accept_conf, "review": t.review_conf, "groups": grp,
                               "strata": per, "rows": rows}, indent=1), encoding="utf-8")
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
