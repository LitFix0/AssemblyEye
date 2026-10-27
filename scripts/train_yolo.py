import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _git(*args):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_absolute_data_yaml(dataset: Path) -> Path:
    src = dataset / "data.yaml"
    if not src.is_file():
        raise FileNotFoundError(f"{src} not found - run export_yolo_dataset.py first")
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    for key in ("train", "val", "names"):
        if key not in cfg:
            raise ValueError(f"{src}: missing '{key}'")
    for key in ("train", "val", "test"):
        if key in cfg and not (dataset / cfg[key]).is_dir():
            raise FileNotFoundError(f"{dataset / cfg[key]} not found")
    cfg = {"path": str(dataset.resolve()), **cfg}
    dst = dataset / "data_abs.yaml"
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ST-03: train a YOLO detector on an exported dataset.")
    ap.add_argument("--dataset", required=True, help="e.g. data/processed/yolo_screw_bag")
    ap.add_argument("--name", required=True, help="run name, e.g. smoke_screw_bag")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0 if platform.system() == "Windows" else 2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    try:
        dataset = Path(args.dataset)
        data_yaml = write_absolute_data_yaml(dataset)
        import cv2
        import torch
        import ultralytics
        from ultralytics import YOLO
    except (OSError, ValueError, yaml.YAMLError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    cuda = torch.cuda.is_available()
    device = ("0" if cuda else "cpu") if args.device == "auto" else args.device
    print(f"device={device}  ultralytics={ultralytics.__version__}  torch={torch.__version__}")

    model = YOLO(args.model)
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, workers=args.workers, seed=args.seed,
                project=str(Path(args.project).resolve()), name=args.name, plots=True)
    save_dir = Path(model.trainer.save_dir)

    report_path = dataset / "export_report.json"
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "args": vars(args),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "platform": platform.platform(),
        "versions": {"python": platform.python_version(), "ultralytics": ultralytics.__version__,
                     "torch": torch.__version__, "cuda": torch.version.cuda, "opencv": cv2.__version__},
        "pretrained_weights": {"name": args.model, "sha256": _sha256(args.model)},
        "dataset_export_report": (json.loads(report_path.read_text(encoding="utf-8"))
                                  if report_path.is_file() else None),
    }
    (save_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\nRun directory: {save_dir}")
    print(f"Best weights:  {save_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())