"""Parse labelme annotations (rectangle, polygon, circle) into boxes and convert them to YOLO lines."""
import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path

MIN_SIDE_PX = 2.0
CLIP_TOLERANCE_PX = 5.0
SUPPORTED_SHAPES = ("rectangle", "polygon", "circle")


class AnnotationError(ValueError):
    """A labelme file is unreadable, inconsistent, or breaks the labelling rules."""


@dataclass(frozen=True)
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    source_shape: str = "rectangle"


def _extent(shape_type, pts, tag):
    """Tight axis-aligned bounds (x1, y1, x2, y2) of a labelme shape."""
    try:
        xs, ys = [float(q[0]) for q in pts], [float(q[1]) for q in pts]
    except (TypeError, ValueError, IndexError) as e:
        raise AnnotationError(f"{tag}: malformed points ({e!r})") from e
    if shape_type == "circle":
        if len(pts) != 2:
            raise AnnotationError(f"{tag}: circle needs exactly 2 points (centre and a point on the edge)")
        r = hypot(xs[1] - xs[0], ys[1] - ys[0])
        return xs[0] - r, ys[0] - r, xs[0] + r, ys[0] + r
    if shape_type == "polygon" and len(pts) < 3:
        raise AnnotationError(f"{tag}: polygon needs at least 3 points")
    if shape_type == "rectangle" and len(pts) < 2:
        raise AnnotationError(f"{tag}: rectangle needs at least 2 points")
    return min(xs), min(ys), max(xs), max(ys)


def parse_labelme(path, classes, image_size=None) -> list[Box]:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        w, h = int(data["imageWidth"]), int(data["imageHeight"])
        shapes = data["shapes"]
    except (OSError, json.JSONDecodeError) as e:
        raise AnnotationError(f"{p.name}: cannot read JSON ({e})") from e
    except (KeyError, TypeError, ValueError) as e:
        raise AnnotationError(f"{p.name}: missing or invalid field ({e!r})") from e
    if image_size is not None and (w, h) != tuple(image_size):
        raise AnnotationError(
            f"{p.name}: JSON says {w}x{h} but the image is {image_size[0]}x{image_size[1]}")
    if not shapes:
        raise AnnotationError(f"{p.name}: no shapes (empty annotation)")

    boxes = []
    for i, s in enumerate(shapes):
        tag = f"{p.name} shape {i}"
        shape_type = s.get("shape_type")
        if shape_type not in SUPPORTED_SHAPES:
            raise AnnotationError(
                f"{tag}: shape_type {shape_type!r} is not supported (use rectangle, polygon or circle)")
        label = s.get("label")
        if label not in classes:
            raise AnnotationError(f"{tag}: unknown label {label!r}")
        x1, y1, x2, y2 = _extent(shape_type, s.get("points") or [], tag)
        tol = CLIP_TOLERANCE_PX
        if x1 < -tol or y1 < -tol or x2 > w + tol or y2 > h + tol:
            raise AnnotationError(f"{tag}: box ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) is outside the image {w}x{h}")
        x1, y1, x2, y2 = max(x1, 0.0), max(y1, 0.0), min(x2, float(w)), min(y2, float(h))
        if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
            raise AnnotationError(f"{tag}: box is too small ({x2 - x1:.1f}x{y2 - y1:.1f} px)")
        boxes.append(Box(label, x1, y1, x2, y2, shape_type))
    return boxes


def to_yolo_line(box: Box, class_index: int, width: int, height: int) -> str:
    cx = (box.x1 + box.x2) / 2 / width
    cy = (box.y1 + box.y2) / 2 / height
    bw = (box.x2 - box.x1) / width
    bh = (box.y2 - box.y1) / height
    return f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def is_unreviewed(path) -> bool:
    """True for model-drafted files whose 'needs_review' flag is still ticked."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False  # unreadable files are reported by parse_labelme
    flags = data.get("flags")
    return isinstance(flags, dict) and flags.get("needs_review") is True