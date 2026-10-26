import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.image_io import ImageValidationError, save_image  # noqa: E402
from app.schemas.spec import load_spec  # noqa: E402
from app.services.detection.fixture_detector import DETECTOR_NAME, load_fixture  # noqa: E402
from app.services.pipeline import run_inspection  # noqa: E402
from app.services.visualization.draw import draw_result  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ST-01 baseline: image + simulated detections -> verdict.")
    ap.add_argument("image")
    ap.add_argument("--spec", default="configs/products/screw_bag.yaml")
    ap.add_argument("--detections", required=True, help="fixture JSON (simulated detector output)")
    ap.add_argument("--out-dir", default="reports/baseline")
    args = ap.parse_args(argv)

    try:
        spec = load_spec(args.spec)
        detections = load_fixture(args.detections)
        result, loaded = run_inspection(args.image, spec, detections, DETECTOR_NAME)
    except (ImageValidationError, ValueError, OSError, KeyError, yaml.YAMLError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.image).stem}__{Path(args.detections).stem}"
    json_path, png_path = out_dir / f"{stem}.json", out_dir / f"{stem}.png"
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    save_image(png_path, draw_result(loaded.bgr, result, spec))

    print(f"{result.status.value}  {spec.product}  ({loaded.name})  findings: {len(result.findings)}")
    for f in result.findings:
        print(f"  [{f.status.value}] {f.code}: {f.message}")
    print("wrote", json_path)
    print("wrote", png_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())