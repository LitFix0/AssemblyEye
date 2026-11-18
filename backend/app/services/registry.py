"""Holds one spec and one lazily-loaded detector per product. Model loading is the only slow,
failure-prone step, so it happens on first use and failures surface as a clean 503, not a crash."""
import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.api.errors import ApiError
from app.core.settings import Settings
from app.schemas.spec import ProductSpec, load_spec


@dataclass
class ProductEntry:
    name: str
    spec: ProductSpec
    weights: Path
    imgsz: int
    suppress: bool
    detector: object | None = None
    load_lock: threading.Lock = field(default_factory=threading.Lock)
    run_lock: threading.Lock = field(default_factory=threading.Lock)  # one inference at a time per model


def default_factory(entry: ProductEntry):
    from app.services.detection.yolo_detector import YoloDetector
    return YoloDetector(entry.weights, entry.imgsz, min_conf=min(0.10, entry.spec.min_review_conf))


class DetectorRegistry:
    def __init__(self, settings: Settings, factory=default_factory):
        self._factory = factory
        self.entries: dict[str, ProductEntry] = {}
        for name, p in settings.serving.products.items():
            spec = load_spec(settings.resolve(p.spec))  # a broken spec stops startup, by design
            self.entries[name] = ProductEntry(name, spec, settings.resolve(p.weights), p.imgsz,
                                              p.cross_class_suppression)

    def get(self, product: str) -> ProductEntry:
        entry = self.entries.get(product)
        if entry is None:
            raise ApiError(404, "unknown_product",
                           f"unknown product {product!r}; available: {sorted(self.entries)}")
        return entry

    def detector_for(self, entry: ProductEntry):
        with entry.load_lock:
            if entry.detector is None:
                try:
                    entry.detector = self._factory(entry)
                except (FileNotFoundError, ImportError) as e:
                    raise ApiError(503, "detector_unavailable", f"detector for {entry.name} cannot load: {e}")
        return entry.detector