"""Check the splicing RULES (not the detector) on your hand-annotated normal images.

Ground-truth labelme boxes are fed to the verification rules as if they were 100%-confident
detections, with the real image pixels. Every annotated image is a normal assembly, so every one
should PASS. Anything else means the spec (hue ranges, gap tolerance) or an annotation needs a look.
Run this before trusting a cable-colour or cable-gap FAIL from the detector-based evaluation.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.image_io import ImageValidationError, load_image  # noqa: E402
from app.datasets.labelme_io import AnnotationError, is_unreviewed, parse_labelme  # noqa: E402
from app.datasets.prepare import read_manifest  # noqa: E402
from app.schemas.inspection import BBox, Detection  # noqa: E402
from app.schemas.spec import load_spec  # noqa: E402
from app.services.verification.color import read_cable_color  # noqa: E402
from app.services.verification.engine import verify_product  # noqa: E402
from app.services.verification.splicing import box_gap  # noqa: E402

CATEGORY = "splicing_connectors"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the splicing rules on ground-truth boxes of annotated normals.")
    ap.add_argument("--annotation-dir", default="data/annotation")
    ap.add_argument("--spec", default=f"configs/products/{CATEGORY}.yaml")
    args = ap.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        if spec.splicing is None:
            raise ValueError(f"{args.spec} has no splicing rules")
        cat_dir = Path(args.annotation_dir) / CATEGORY
        classes = [c.strip() for c in (cat_dir / "classes.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
        manifest = read_manifest(cat_dir / "manifest.csv")
        if not manifest:
            raise ValueError(f"no manifest at {cat_dir / 'manifest.csv'}")
    except (ValueError, OSError, yaml.YAMLError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    rules = spec.splicing
    rows, skipped = [], []
    for m in manifest:
        stem = Path(m["file"]).stem
        label_path = cat_dir / "labels_json" / f"{stem}.json"
        if not label_path.is_file():
            skipped.append((m["file"], "no label file"))
            continue
        if is_unreviewed(label_path):
            skipped.append((m["file"], "label still marked needs_review"))
            continue
        try:
            img = load_image(cat_dir / "images" / m["file"])
            boxes = parse_labelme(label_path, classes, image_size=(img.width, img.height))
        except (ImageValidationError, AnnotationError) as e:
            skipped.append((m["file"], str(e)))
            continue
        dets = [Detection(class_name=b.label, confidence=1.0, bbox=BBox(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2))
                for b in boxes]
        cables = [d for d in dets if d.class_name == rules.cable_class]
        conns = sorted((d for d in dets if d.class_name in rules.connector_classes), key=lambda d: d.bbox.x1)
        status, findings = verify_product(dets, spec, img.bgr)
        row = {"file": m["file"], "role": m.get("role", ""), "verdict": status.value,
               "findings": [f"{f.code}" for f in findings], "messages": [f.message for f in findings],
               "type": conns[0].class_name if len(conns) == 2 and conns[0].class_name == conns[1].class_name else "?"}
        if len(cables) == 1 and len(conns) == 2:
            row["gaps"] = [box_gap(cables[0].bbox, c.bbox) for c in conns]
            rd = read_cable_color(img.bgr, cables[0].bbox, [c.bbox for c in conns], rules.color)
            row["color"], row["share"], row["color_reason"] = rd.name, rd.share, rd.reason
            row["expected"] = rules.cable_colors.get(row["type"])
            row["diag"] = (img.width ** 2 + img.height ** 2) ** 0.5
        rows.append(row)

    print(f"\n{CATEGORY}: rules on ground-truth boxes, {len(rows)} annotated normal images "
          f"({len(skipped)} skipped)")
    print(f"spec: {spec.product} v{spec.version}; gap tolerance {rules.end_tolerance:g} of the diagonal\n")
    print(f"{'file':<16}{'type':<13}{'verdict':<8}{'colour read':<18}{'expected':<9}{'gaps px (L,R)':<16}findings")
    for r in rows:
        col = "-" if "color" not in r else (f"{r['color']} {r['share']:.0%}" if r["color"] else "UNKNOWN")
        gaps = "-" if "gaps" not in r else f"{r['gaps'][0]:.1f}, {r['gaps'][1]:.1f}"
        print(f"{r['file']:<16}{r['type']:<13}{r['verdict']:<8}{col:<18}{str(r.get('expected', '-')):<9}"
              f"{gaps:<16}{' '.join(r['findings'])}")

    if not rows:
        print("\nno usable annotated images found")
        for f, why in skipped:
            print(f"  skipped {f}: {why}")
        return 2
    verdicts = Counter(r["verdict"] for r in rows)
    print(f"\nverdicts: {dict(verdicts)} (all annotated images are normal, so every one should be PASS)")
    by_type: dict[str, Counter] = {}
    for r in rows:
        if "color" in r:
            by_type.setdefault(r["type"], Counter())[r["color"] or "UNKNOWN"] += 1
    print("cable colour read, by connector type (expected: " +
          ", ".join(f"{k}={v}" for k, v in rules.cable_colors.items()) + "):")
    for t in sorted(by_type):
        print(f"   {t:<13}{dict(by_type[t])}")
    gaps = [g for r in rows if "gaps" in r for g in r["gaps"]]
    if gaps:
        diag = next(r["diag"] for r in rows if "diag" in r)
        print(f"cable-to-connector gap: max {max(gaps):.1f} px; tolerance {rules.end_tolerance * diag:.1f} px")
    bad = [r for r in rows if r["verdict"] != "PASS"]
    if bad:
        print("\nnot PASS (look at these images, then at the spec):")
        for r in bad[:15]:
            print(f"  {r['file']}: " + " | ".join(r["messages"]))
    for f, why in skipped:
        print(f"skipped {f}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
