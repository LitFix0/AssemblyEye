"""Dump the detections behind selected rows of an eval_defects report (splicing only).

Answers: what ARE the extra low-confidence cable boxes (a second box on the same cable, or something
else), and why did a defect image PASS? Read-only: no thresholds or rules are changed. DEV split only.

  --select normal-review : DEV normals that were flagged (REVIEW/FAIL)
  --select defect-noise  : defect images flagged only by cable-confidence findings
  --select defect-pass   : defect images that PASSED (missed)
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.image_io import load_image  # noqa: E402
from app.datasets.loco import DatasetLayoutError, index_category  # noqa: E402
from app.evaluation.counts import iou  # noqa: E402
from app.schemas.inspection import BBox  # noqa: E402
from app.schemas.spec import load_spec  # noqa: E402
from app.services.detection.yolo_detector import YoloDetector  # noqa: E402
from app.services.verification.color import read_cable_color  # noqa: E402
from app.services.verification.count_rules import band_of  # noqa: E402
from app.services.verification.splicing import box_gap  # noqa: E402

CATEGORY = "splicing_connectors"
NOISE = {"UNCERTAIN_EXTRA:cable", "UNCERTAIN_PART:cable"}


def overlap_of_smaller(a: BBox, b: BBox) -> float:
    """Intersection area / area of the smaller box: 1.0 means one box lies entirely inside the other."""
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    area = lambda b_: max(1e-9, (b_.x2 - b_.x1) * (b_.y2 - b_.y1))  # noqa: E731
    return ix * iy / min(area(a), area(b))


def fmt_box(b: BBox) -> str:
    return f"({b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f})"


def select_rows(rows, how):
    if how == "normal-review":
        return [r for r in rows if r["group"] == "normal" and r["verdict"] != "PASS"]
    if how == "defect-noise":
        return [r for r in rows if r["group"] != "normal" and r["findings"] and set(r["findings"]) <= NOISE]
    if how == "defect-pass":
        return [r for r in rows if r["group"] != "normal" and r["verdict"] == "PASS"]
    return [r for r in rows if (r["group"] == "normal" and r["verdict"] != "PASS")
            or (r["group"] != "normal" and (r["verdict"] == "PASS" or set(r["findings"]) <= NOISE))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="reports/eval_defects_splicing_connectors_*.json")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--select", default="all", choices=["all", "normal-review", "defect-noise", "defect-pass"])
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--spec", default=f"configs/products/{CATEGORY}.yaml")
    ap.add_argument("--root", default="data/raw/mvtec_loco")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        if spec.splicing is None:
            raise ValueError("spec has no splicing rules")
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        if report.get("category") != CATEGORY or report.get("split") != "dev":
            raise ValueError("report must be a splicing_connectors DEV report")
        rows = select_rows(report["rows"], args.select)[: args.max]
        samples = {s.rel_path: s for s in index_category(args.root, CATEGORY)}
        detector = YoloDetector(args.weights, args.imgsz, min_conf=min(0.10, spec.min_review_conf),
                                device=args.device)
    except (DatasetLayoutError, ValueError, OSError, ImportError, KeyError, yaml.YAMLError,
            json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    rules = spec.splicing
    print(f"{CATEGORY}: {len(rows)} image(s), select={args.select}, detector {detector.name}")
    print(f"cable thresholds: accept {spec.thresholds_for(rules.cable_class).accept_conf:g}, "
          f"review {spec.thresholds_for(rules.cable_class).review_conf:g}\n")
    for r in rows:
        sample = samples.get(r["image"])
        if sample is None:
            print(f"{r['image']}: not found in dataset\n")
            continue
        img = load_image(sample.image_path)
        diag = (img.width ** 2 + img.height ** 2) ** 0.5
        dets = sorted((d for d in detector.detect(img.bgr) if band_of(d, spec) != "ignored"),
                      key=lambda d: (d.class_name != rules.cable_class, -d.confidence))
        print(f"{r['image']}  [{r['stratum']}]  {r['verdict']}  {' '.join(r['findings']) or '-'}")
        print(f"  image {img.width}x{img.height}, diagonal {diag:.0f} px, "
              f"tolerance {rules.end_tolerance * diag:.0f} px")
        cables = [d for d in dets if d.class_name == rules.cable_class]
        conns = sorted((d for d in dets if d.class_name in rules.connector_classes), key=lambda d: -d.confidence)
        for d in conns + cables:
            print(f"  {d.class_name:<12}{d.confidence:>5.2f}  {band_of(d, spec):<10}{fmt_box(d.bbox)}")
        if cables:
            top = cables[0]
            for d in cables[1:]:
                dist = ((((d.bbox.x1 + d.bbox.x2) - (top.bbox.x1 + top.bbox.x2)) / 2) ** 2
                        + (((d.bbox.y1 + d.bbox.y2) - (top.bbox.y1 + top.bbox.y2)) / 2) ** 2) ** 0.5
                print(f"    cable {d.confidence:.2f} vs top cable {top.confidence:.2f}: "
                      f"IoU {iou(top.bbox, d.bbox):.2f}, overlap-of-smaller {overlap_of_smaller(top.bbox, d.bbox):.2f}, "
                      f"centre distance {dist / diag:.1%} of diagonal")
            if len(conns) >= 2:
                two = sorted(conns[:2], key=lambda d: d.bbox.x1)
                gaps = [box_gap(top.bbox, c.bbox) for c in two]
                rd = read_cable_color(img.bgr, top.bbox, [c.bbox for c in two], rules.color)
                col = f"{rd.name} {rd.share:.0%}" if rd.name else f"UNKNOWN ({rd.reason})"
                types = "/".join(sorted({c.class_name for c in two}))
                print(f"    top cable: gaps to connectors {gaps[0]:.0f} / {gaps[1]:.0f} px; colour {col}; "
                      f"connector type {types} (needs {rules.cable_colors.get(two[0].class_name, '?')})")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())