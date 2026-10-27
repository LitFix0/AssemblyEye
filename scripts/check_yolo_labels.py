import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.image_io import load_image, save_image  # noqa: E402
from app.datasets.yolo_export import read_yolo_labels  # noqa: E402

PALETTE = [(255, 128, 0), (0, 200, 0), (0, 0, 255), (255, 0, 255), (0, 200, 200), (200, 200, 0)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Round-trip check of an exported YOLO dataset.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="reports/label_checks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args(argv)
    ds = Path(args.dataset)
    try:
        names = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))["names"]
        items = []
        n_img, n_box = Counter(), Counter()
        sides = defaultdict(lambda: defaultdict(lambda: ([], [])))  # split -> class -> (orig, scaled)
        for split_dir in sorted((ds / "images").iterdir()):
            for img in sorted(split_dir.glob("*.png")):
                lbl = ds / "labels" / split_dir.name / f"{img.stem}.txt"
                if not lbl.is_file():
                    raise ValueError(f"missing label file for {img.name}")
                li = load_image(img)
                boxes = read_yolo_labels(lbl, li.width, li.height)
                scale = args.imgsz / max(li.width, li.height)
                n_img[split_dir.name] += 1
                n_box[split_dir.name] += len(boxes)
                for c, x1, y1, x2, y2 in boxes:
                    s = min(x2 - x1, y2 - y1)
                    sides[split_dir.name][names[c]][0].append(s)
                    sides[split_dir.name][names[c]][1].append(s * scale)
                items.append((li, boxes, f"{split_dir.name}/{img.name}"))

        for split in n_img:
            print(f"{split}: {n_img[split]} images, {n_box[split]} boxes")
            for cls, (orig, scaled) in sorted(sides[split].items()):
                print(f"  {cls:<12} n={len(orig):>3}  min side px: {min(orig):.0f}/{median(orig):.0f} (min/median)"
                      f"  at imgsz {args.imgsz}: {min(scaled):.0f}/{median(scaled):.0f}")

        tiles = []
        for li, boxes, tag in random.Random(args.seed).sample(items, min(args.n, len(items))):
            canvas = li.bgr.copy()
            for c, x1, y1, x2, y2 in boxes:
                color = PALETTE[c % len(PALETTE)]
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(canvas, names[c], (int(x1), max(24, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, tag, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(canvas, (800, int(canvas.shape[0] * 800 / canvas.shape[1]))))
        rows = []
        for i in range(0, len(tiles), 2):
            pair = tiles[i:i + 2]
            if len(pair) == 1:
                pair.append(pair[0] * 0)
            height = max(t.shape[0] for t in pair)
            pair = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in pair]
            rows.append(cv2.hconcat(pair))
        out_path = Path(args.out) / f"{ds.name}.jpg"
        save_image(out_path, cv2.vconcat(rows))
        print("wrote", out_path)
    except (ValueError, OSError, KeyError, yaml.YAMLError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())