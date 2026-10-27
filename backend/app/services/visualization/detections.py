import zlib

import cv2
import numpy as np

from app.schemas.inspection import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = [(255, 128, 0), (0, 200, 0), (0, 0, 255), (255, 0, 255), (0, 200, 200), (200, 200, 0)]


def draw_detections(bgr: np.ndarray, detections: list[Detection], title: str = "") -> np.ndarray:
    """Raw detector output: boxes with class and confidence. A banner goes ABOVE the image."""
    out = bgr.copy()
    h, w = out.shape[:2]
    s = max(w, h) / 1000.0
    thick, fs = max(1, round(2 * s)), 0.5 * s
    for d in detections:
        color = PALETTE[zlib.crc32(d.class_name.encode()) % len(PALETTE)]
        p1, p2 = (int(d.bbox.x1), int(d.bbox.y1)), (int(d.bbox.x2), int(d.bbox.y2))
        cv2.rectangle(out, p1, p2, color, thick)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(label, FONT, fs, thick)
        y0 = max(0, p1[1] - th - base - 2)
        cv2.rectangle(out, (p1[0], y0), (p1[0] + tw + 4, y0 + th + base + 2), color, -1)
        cv2.putText(out, label, (p1[0] + 2, y0 + th + 1), FONT, fs, (0, 0, 0),
                    max(1, thick - 1), cv2.LINE_AA)
    if not title:
        return out
    banner_h = int(34 * s)
    banner = np.full((banner_h, w, 3), 25, dtype=np.uint8)
    cv2.putText(banner, title[:150], (int(8 * s), int(banner_h * 0.7)), FONT, fs * 1.1,
                (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    return np.vstack([banner, out])