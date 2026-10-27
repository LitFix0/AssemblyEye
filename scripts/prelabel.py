import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.image_io import ImageValidationError, load_image  # noqa: E402
from app.datasets.labelme_io import AnnotationError, parse_labelme  # noqa: E402
from app.datasets.prelabel import build_labelme, cap_to_expected  # noqa: E402
from app.datasets.prepare import read_manifest  # noqa: E402
from app.evaluation.counts import suppress_cross_class  # noqa: E402
from app.schemas.spec import load_spec  # noqa: E402
from app.services.detection.yolo_detector import YoloDetector  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write labelme DRAFTS from a detector, for human review.")
    ap.add_argument("--category", required=True, choices=["screw_bag", "splicing_connectors"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--annotation-dir", default="data/annotation")
    ap.add_argument("--classes", default="configs/datasets/loco_classes.yaml")
    ap.add_argument("--spec", default="configs/products/screw_bag.yaml")
    ap.add_argument("--no-cap", action="store_true",
                    help="keep every suppressed detection instead of capping to the expected composition")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    try:
        classes = yaml.safe_load(Path(args.classes).read_text(encoding="utf-8"))[args.category]
        cat = Path(args.annotation_dir) / args.category
        labels = cat / "labels_json"
        rows = read_manifest(cat / "manifest.csv")
        if not rows:
            raise ValueError(f"no manifest in {cat}")
        templates = sorted(labels.glob("*.json"))
        if not templates:
            raise ValueError("label at least one image by hand first (its JSON is the format template)")
        template = json.loads(templates[0].read_text(encoding="utf-8"))
        required = load_spec(args.spec).required_counts if args.category == "screw_bag" else None
        detector = YoloDetector(args.weights, args.imgsz, min_conf=0.10, device=args.device)
    except (OSError, ValueError, KeyError, ImportError, yaml.YAMLError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    written, skipped, empty, bad = [], 0, [], []
    for r in rows:
        name = r["file"]
        out = labels / f"{Path(name).stem}.json"
        if out.exists():  # never overwrite a human's work
            skipped += 1
            continue
        try:
            img = load_image(cat / "images" / name)
        except ImageValidationError as e:
            bad.append(f"{name}: {e}")
            continue
        dets = suppress_cross_class(detector.detect(img.bgr))
        if not args.no_cap:
            dets = cap_to_expected(dets, args.category, required)
        if not dets:
            empty.append(name)
            continue
        out.write_text(json.dumps(build_labelme(template, name, img.width, img.height, dets), indent=2),
                       encoding="utf-8")
        try:  # our own parser must accept what we just wrote
            parse_labelme(out, classes, (img.width, img.height))
        except AnnotationError as e:
            out.unlink()
            bad.append(str(e))
            continue
        written.append(name)

    print(f"{args.category}: {len(written)} drafts written, {skipped} already labelled (kept), "
          f"{len(empty)} with no detections (label by hand), {len(bad)} problems")
    for line in bad:
        print("  PROBLEM:", line)
    if empty:
        print("  no detections:", ", ".join(empty[:10]))
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())