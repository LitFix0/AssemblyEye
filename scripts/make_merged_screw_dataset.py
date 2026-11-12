import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.datasets.variants import merge_classes, remap_label_lines  # noqa: E402

MERGE = {"screw_long": "screw", "screw_short": "screw"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dataset variant with screw_long + screw_short merged into 'screw'.")
    ap.add_argument("--src", default="data/processed/yolo_screw_bag")
    ap.add_argument("--dst", default="data/processed/yolo_screw_bag_1cls")
    args = ap.parse_args(argv)
    src, dst = Path(args.src), Path(args.dst)
    try:
        cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
        report = json.loads((src / "export_report.json").read_text(encoding="utf-8"))
        new_names, mapping = merge_classes(cfg["names"], MERGE)
        if dst.exists() and any(dst.iterdir()):
            if not (dst / "data.yaml").is_file():
                raise ValueError(f"refusing to clear {dst}: it does not look like a previous export")
            shutil.rmtree(dst)
        n_img = n_box = 0
        for lbl in sorted((src / "labels").rglob("*.txt")):
            split = lbl.parent.name
            (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
            (dst / "images" / split).mkdir(parents=True, exist_ok=True)
            text = remap_label_lines(lbl.read_text(encoding="utf-8"), mapping)
            (dst / "labels" / split / lbl.name).write_text(text, encoding="utf-8")
            shutil.copy2(src / "images" / split / f"{lbl.stem}.png",
                         dst / "images" / split / f"{lbl.stem}.png")
            n_img += 1
            n_box += len(text.splitlines())
        out_cfg = {k: v for k, v in cfg.items() if k != "names"}
        out_cfg["names"] = dict(enumerate(new_names))
        (dst / "data.yaml").write_text(yaml.safe_dump(out_cfg, sort_keys=False), encoding="utf-8")
        boxes = {}
        for cls, n in report["boxes_per_class"].items():
            target = MERGE.get(cls, cls)
            boxes[target] = boxes.get(target, 0) + n
        report.update({"classes": new_names, "boxes_per_class": boxes,
                       "variant": {"merged_classes": MERGE, "source_dataset": str(src)}})
        (dst / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (OSError, KeyError, ValueError, yaml.YAMLError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"{dst}: {n_img} images, {n_box} boxes, classes {new_names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())