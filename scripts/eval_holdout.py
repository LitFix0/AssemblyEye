import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.detection.yolo_detector import file_fingerprint  # noqa: E402


def holdout_yaml(dataset: Path) -> Path:
    cfg = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    if "test" not in cfg:
        raise ValueError(f"{dataset / 'data.yaml'} has no 'test' entry (the holdout split)")
    if not (dataset / cfg["test"]).is_dir():
        raise FileNotFoundError(f"{dataset / cfg['test']} not found")
    dst = dataset / "data_abs_holdout.yaml"
    dst.write_text(yaml.safe_dump({"path": str(dataset.resolve()), **cfg}, sort_keys=False), encoding="utf-8")
    return dst


def count_instances(dataset: Path, names: dict):
    files = sorted((dataset / "labels" / "holdout").glob("*.txt"))
    counts = Counter()
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[names[int(line.split()[0])]] += 1
    return len(files), counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Once-only box-level metrics on the holdout split.")
    ap.add_argument("--dataset", required=True, help="e.g. data/processed/yolo_screw_bag")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--again", action="store_true", help="re-run; the previous result is backed up")
    args = ap.parse_args(argv)
    ds, stem = Path(args.dataset), Path(args.weights).stem
    out = Path("reports") / f"holdout_{ds.name}_{stem}.json"
    try:
        if out.exists() and not args.again:
            raise ValueError(f"{out} exists: the holdout was already evaluated for these weights "
                             "(use --again only with a reason, and say so in the report)")
        data_yaml = holdout_yaml(ds)
        names = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))["names"]
        n_img, inst = count_instances(ds, names)
        if n_img == 0:
            raise ValueError("no holdout labels found")
        from ultralytics import YOLO
        model = YOLO(args.weights)
        m = model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=4, device=args.device,
                      workers=0, plots=False, verbose=False, project=str(Path("runs").resolve()),
                      name=f"holdout_{stem}", exist_ok=True)
        box = m.box
        per = {}
        for i, c in enumerate(box.ap_class_index):
            p, r, ap50, ap = box.class_result(i)
            per[m.names[int(c)]] = {"boxes": inst.get(m.names[int(c)], 0), "precision": float(p),
                                    "recall": float(r), "map50": float(ap50), "map50_95": float(ap)}
        overall = {"boxes": sum(inst.values()), "precision": float(box.mp), "recall": float(box.mr),
                   "map50": float(box.map50), "map50_95": float(box.map)}
    except (ValueError, OSError, KeyError, ImportError, AttributeError, yaml.YAMLError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    out.parent.mkdir(exist_ok=True)
    rerun = out.exists()
    if rerun:
        out.rename(out.with_name(out.name + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    out.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dataset": str(ds),
        "weights": args.weights, "weights_fingerprint": file_fingerprint(args.weights), "imgsz": args.imgsz,
        "holdout_images": n_img, "rerun": rerun, "overall": overall, "per_class": per}, indent=2),
        encoding="utf-8")
    print(f"\nholdout: {n_img} images, {overall['boxes']} boxes   weights @{file_fingerprint(args.weights)}")
    print(f"{'class':<14}{'boxes':>6}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for name, v in {**per, "ALL": overall}.items():
        print(f"{name:<14}{v['boxes']:>6}{v['precision']:>8.3f}{v['recall']:>8.3f}"
              f"{v['map50']:>8.3f}{v['map50_95']:>10.3f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())