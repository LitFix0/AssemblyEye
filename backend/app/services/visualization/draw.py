import cv2
import numpy as np

from app.schemas.inspection import InspectionResult, Status
from app.schemas.spec import ProductSpec

FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_BGR = {Status.PASS: (60, 160, 40), Status.FAIL: (40, 40, 200), Status.REVIEW: (0, 140, 230)}
CONFIDENT_BGR = (200, 160, 0)
UNCERTAIN_BGR = (0, 165, 255)
IGNORED_BGR = (150, 150, 150)


def draw_result(bgr: np.ndarray, result: InspectionResult, spec: ProductSpec) -> np.ndarray:
    """Boxes are drawn on a copy of the image; a banner is added ABOVE it so no evidence is hidden."""
    out = bgr.copy()
    h, w = out.shape[:2]
    s = max(w, h) / 1000.0
    thick = max(1, round(2 * s))
    fs = 0.45 * s
    t = spec.thresholds

    for d in result.detections:
        if d.confidence >= t.accept_conf:
            color = CONFIDENT_BGR
        elif d.confidence >= t.review_conf:
            color = UNCERTAIN_BGR
        else:
            color = IGNORED_BGR
        p1, p2 = (int(d.bbox.x1), int(d.bbox.y1)), (int(d.bbox.x2), int(d.bbox.y2))
        cv2.rectangle(out, p1, p2, color, thick)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(label, FONT, fs, thick)
        y0 = max(0, p1[1] - th - base - 2)
        cv2.rectangle(out, (p1[0], y0), (p1[0] + tw + 4, y0 + th + base + 2), color, -1)
        cv2.putText(out, label, (p1[0] + 2, y0 + th + 1), FONT, fs, (0, 0, 0),
                    max(1, thick - 1), cv2.LINE_AA)

    lines = [f"{result.status.value} | {result.product} | {result.image.name}"]
    lines += [f"[{f.status.value}] {f.message}" for f in result.findings]
    lines.append(f"detector: {result.detector}")

    line_h, pad = int(26 * s), int(8 * s)
    head_h = 2 * pad + line_h
    banner = np.full((2 * pad + line_h * len(lines), w, 3), 25, dtype=np.uint8)
    cv2.rectangle(banner, (0, 0), (w, head_h), STATUS_BGR[result.status], -1)
    for i, line in enumerate(lines):
        scale = fs * 1.25 if i == 0 else fs
        y = pad + line_h * (i + 1) - int(6 * s)
        cv2.putText(banner, line[:140], (pad, y), FONT, scale, (255, 255, 255),
                    max(1, thick - 1), cv2.LINE_AA)
    return np.vstack([banner, out])