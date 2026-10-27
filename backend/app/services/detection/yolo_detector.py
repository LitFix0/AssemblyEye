"""YOLO detector wrapper: the only module that touches Ultralytics.

Ultralytics is imported lazily so the rest of the application and the unit tests
do not need it installed.
"""
import hashlib
from pathlib import Path

import numpy as np

from app.schemas.inspection import BBox, Detection


def file_fingerprint(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def to_detections(xyxy, confs, cls_ids, names, width, height) -> list[Detection]:
    """Convert raw model output to schema objects: clamp to the image, drop degenerate boxes,
    sort by confidence."""
    out = []
    for (x1, y1, x2, y2), conf, c in zip(xyxy, confs, cls_ids):
        x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
        x2, y2 = min(float(width), float(x2)), min(float(height), float(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(Detection(class_name=names[int(c)],
                             confidence=min(1.0, max(0.0, float(conf))),
                             bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2)))
    out.sort(key=lambda d: -d.confidence)
    return out


class YoloDetector:
    def __init__(self, weights, imgsz: int = 640, min_conf: float = 0.10,
                 iou: float = 0.7, device=None):
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"weights not found: {self.weights}")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics is not installed (pip install ultralytics)") from e
        self.imgsz, self.min_conf, self.iou, self.device = imgsz, min_conf, iou, device
        self.fingerprint = file_fingerprint(self.weights)
        self._model = YOLO(str(self.weights))

    @property
    def name(self) -> str:
        return (f"yolo:{self.weights.name}@{self.fingerprint} "
                f"imgsz={self.imgsz} min_conf={self.min_conf}")

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        result = self._model.predict(bgr, imgsz=self.imgsz, conf=self.min_conf, iou=self.iou,
                                     device=self.device, verbose=False)[0]
        boxes = result.boxes
        h, w = bgr.shape[:2]
        return to_detections(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                             boxes.cls.cpu().numpy(), result.names, w, h)