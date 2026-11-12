from math import hypot

from app.schemas.inspection import Detection


def diagonal(d: Detection) -> float:
    return hypot(d.bbox.x2 - d.bbox.x1, d.bbox.y2 - d.bbox.y1)


def fit_length_threshold(longs: list[float], shorts: list[float]) -> dict:
    """Pick a diagonal threshold (px) separating long from short screws.
    If the classes separate completely, use the midpoint of the gap (maximum margin); otherwise the
    threshold with the most correct screws. Fit on TRAIN/VAL labels only, never on dev or holdout."""
    if not longs or not shorts:
        raise ValueError("need at least one long and one short screw label")
    total = len(longs) + len(shorts)
    lo_long, hi_short = min(longs), max(shorts)
    if hi_short < lo_long:
        return {"threshold": (hi_short + lo_long) / 2, "separable": True,
                "margin_px": lo_long - hi_short, "wrong": 0, "total": total}
    best_t, best_ok = None, -1
    for t in sorted(longs + shorts):
        ok = sum(v >= t for v in longs) + sum(v < t for v in shorts)
        if ok > best_ok:
            best_t, best_ok = t, ok
    return {"threshold": best_t, "separable": False, "margin_px": 0.0,
            "wrong": total - best_ok, "total": total}


def relabel_by_length(detections, threshold: float, merged="screw",
                      long_name="screw_long", short_name="screw_short"):
    out = []
    for d in detections:
        if d.class_name == merged:
            name = long_name if diagonal(d) >= threshold else short_name
            out.append(d.model_copy(update={"class_name": name}))
        else:
            out.append(d)
    return out