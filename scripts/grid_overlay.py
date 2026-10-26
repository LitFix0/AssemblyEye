import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.image_io import load_image, save_image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw a labelled coordinate grid on an image.")
    ap.add_argument("image")
    ap.add_argument("--out", default="reports/grid.png")
    ap.add_argument("--step", type=int, default=100)
    a = ap.parse_args()

    img = load_image(a.image)
    out = img.bgr.copy()
    h, w = out.shape[:2]
    yellow = (0, 255, 255)
    for x in range(0, w, a.step):
        cv2.line(out, (x, 0), (x, h - 1), yellow, 1)
        cv2.putText(out, str(x), (x + 3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2, cv2.LINE_AA)
    for y in range(0, h, a.step):
        cv2.line(out, (0, y), (w - 1, y), yellow, 1)
        cv2.putText(out, str(y), (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2, cv2.LINE_AA)
    save_image(a.out, out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()