import cv2
import numpy as np

from app.schemas.inspection import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX
# BGR. Colours are assigned by position in class_order, so two classes never share one
# (the first version hashed class names, and two pairs of classes collided).
PALETTE = [(255, 128, 0), (0, 200, 0), (0, 0, 255), (255, 0, 255),
           (0, 200, 200), (200, 200, 0), (128, 0, 255), (0, 128, 255)]
UNKNOWN_BGR = (150, 150, 150)


def class_color(name: str, class_order: list[str]) -> tuple[int, int, int]:
    if name in class_order:
        return PALETTE[class_order.index(name) % len(PALETTE)]
    return UNKNOWN_BGR


def draw_detections(bgr: np.ndarray, detections: list[Detection], title: str = "",
                    class_order: list[str] | None = None) -> np.ndarray:
    """Raw detector output: boxes with class and confidence. A banner goes ABOVE the image."""
    out = bgr.copy()
    order = class_order if class_order is not None else sorted({d.class_name for d in detections})
    h, w = out.shape[:2]
    s = max(w, h) / 1000.0
    thick, fs = max(1, round(2 * s)), 0.5 * s
    for d in detections:
        color = class_color(d.class_name, order)
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