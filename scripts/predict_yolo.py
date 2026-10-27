import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.image_io import ImageValidationError, load_image, save_image  # noqa: E402
from app.services.detection.yolo_detector import YoloDetector  # noqa: E402
from app.services.visualization.detections import draw_detections  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run a trained YOLO detector on images (detections only, no verdict).")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-conf", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="reports/predictions")
    args = ap.parse_args(argv)
    try:
        detector = YoloDetector(args.weights, args.imgsz, args.min_conf, device=args.device)
    except (FileNotFoundError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for path in args.images:
        try:
            img = load_image(path)
        except ImageValidationError as e:
            print(f"ERROR: {path}: {e}", file=sys.stderr)
            failed = True
            continue
        dets = detector.detect(img.bgr)
        stem = Path(path).stem
        payload = {"image": img.name, "detector": detector.name,
                   "detections": [d.model_dump() for d in dets]}
        (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        save_image(out_dir / f"{stem}.png", draw_detections(img.bgr, dets, title=detector.name, class_order=detector.class_names))
        by_class = defaultdict(list)
        for d in dets:
            by_class[d.class_name].append(d.confidence)
        summary = ", ".join(f"{c}: {len(v)} ({', '.join(f'{x:.2f}' for x in v)})"
                            for c, v in sorted(by_class.items())) or "no detections"
        print(f"{img.name}: {summary}")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())