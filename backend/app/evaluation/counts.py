"""Count-level evaluation of a detector on NORMAL images.

mAP ranks detections and forgives low-confidence extras; the verdict logic does not. What matters for
inspection is whether a normal image produces exactly the expected parts at the operating threshold.
"""
from collections import Counter

from app.schemas.inspection import BBox, Detection


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def counts_at(detections: list[Detection], thr: float) -> Counter:
    return Counter(d.class_name for d in detections if d.confidence >= thr)


def cross_class_pairs(detections: list[Detection], thr: float, iou_thr: float = 0.7) -> int:
    kept = [d for d in detections if d.confidence >= thr]
    return sum(1 for i, a in enumerate(kept) for b in kept[i + 1:]
               if a.class_name != b.class_name and iou(a.bbox, b.bbox) >= iou_thr)


def suppress_cross_class(detections: list[Detection], iou_thr: float = 0.7) -> list[Detection]:
    """Greedy, highest confidence first: drop any box that overlaps an already-kept box of ANOTHER
    class. Same-class duplicates are left alone (that is per-class NMS's job)."""
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda x: -x.confidence):
        if all(not (k.class_name != d.class_name and iou(k.bbox, d.bbox) >= iou_thr) for k in kept):
            kept.append(d)
    return kept


def exact_screw_bag(counts: Counter, required: dict[str, int]) -> bool:
    return all(counts.get(c, 0) == n for c, n in required.items()) and all(c in required for c in counts)


def exact_splicing(counts: Counter) -> bool:
    connectors = {c: n for c, n in counts.items() if c.startswith("connector_")}
    others = [c for c in counts if c != "cable" and not c.startswith("connector_")]
    return (counts.get("cable", 0) == 1 and sum(connectors.values()) == 2
            and len(connectors) == 1 and not others)


def summarize(per_image: list[list[Detection]], check, thresholds, iou_thr: float = 0.7) -> list[dict]:
    rows = []
    for thr in thresholds:
        ok = sum(1 for dets in per_image if check(counts_at(dets, thr)))
        pairs = [cross_class_pairs(dets, thr, iou_thr) for dets in per_image]
        rows.append({"threshold": thr, "exact": ok, "images": len(per_image),
                     "exact_rate": ok / len(per_image) if per_image else 0.0,
                     "images_with_cross_class_overlap": sum(1 for p in pairs if p > 0),
                     "cross_class_pairs": sum(pairs)})
    return rows