from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_BYTES = 25 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 8000


class ImageValidationError(ValueError):
    """The input file is not an acceptable image."""


@dataclass(frozen=True)
class LoadedImage:
    name: str
    bgr: np.ndarray
    width: int
    height: int


def load_image(path) -> LoadedImage:
    p = Path(path)
    if not p.is_file():
        raise ImageValidationError(f"file not found: {p}")
    if p.suffix.lower() not in ALLOWED_EXT:
        raise ImageValidationError(f"unsupported file type: {p.suffix or '(none)'}")
    size = p.stat().st_size
    if size == 0:
        raise ImageValidationError("file is empty")
    if size > MAX_BYTES:
        raise ImageValidationError(f"file too large: {size} bytes (max {MAX_BYTES})")
    try:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is None:
        raise ImageValidationError("file could not be decoded as an image")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise ImageValidationError(
            f"image dimensions {w}x{h} outside allowed range {MIN_SIDE}-{MAX_SIDE} px"
        )
    return LoadedImage(name=p.name, bgr=img, width=w, height=h)


def save_image(path, bgr: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".png", bgr)
    if not ok:
        raise OSError(f"could not encode image for {p}")
    buf.tofile(str(p))