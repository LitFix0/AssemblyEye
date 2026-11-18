"""Read a cable's colour from pixels (colour is NOT a detector class).

Method: take the cable box, blank out the connector boxes (the cable box overlaps them and connector
pixels are not cable), keep only clearly coloured pixels (saturation and brightness above a floor, so
grey background and shadows do not vote), and let those pixels vote by hue. The reading is trusted only
if enough pixels voted and one colour clearly wins; otherwise the answer is 'unknown', which the
verifier turns into REVIEW. The hue ranges and floors are in the spec and are placeholders until checked
on real images (scripts/check_splicing_rules.py).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.schemas.inspection import BBox
from app.schemas.spec import ColorModel


@dataclass(frozen=True)
class ColorReading:
    name: str | None  # None = could not be read reliably
    share: float  # winning colour's share of the saturated pixels (0 if unknown)
    n_pixels: int  # pixels examined (inside the cable box, outside the connector boxes)
    n_saturated: int  # of those, clearly coloured pixels that voted
    shares: dict = field(default_factory=dict)  # colour name -> share of voters
    reason: str = ""  # why it is unknown, empty when name is set


def _clip(b: BBox, w: int, h: int) -> tuple[int, int, int, int]:
    return (max(0, int(np.floor(b.x1))), max(0, int(np.floor(b.y1))),
            min(w, int(np.ceil(b.x2))), min(h, int(np.ceil(b.y2))))


def read_cable_color(bgr: np.ndarray, cable: BBox, exclude: list[BBox], model: ColorModel) -> ColorReading:
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("expected an 8-bit BGR image")
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _clip(cable, w, h)
    if x2 <= x1 or y2 <= y1:
        return ColorReading(None, 0.0, 0, 0, reason="cable box is outside the image")

    keep = np.ones((y2 - y1, x2 - x1), dtype=bool)
    for e in exclude:
        ex1, ey1, ex2, ey2 = _clip(e, w, h)
        ix1, iy1, ix2, iy2 = max(x1, ex1), max(y1, ey1), min(x2, ex2), min(y2, ey2)
        if ix2 > ix1 and iy2 > iy1:
            keep[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = False
    n_pixels = int(keep.sum())
    if n_pixels == 0:
        return ColorReading(None, 0.0, 0, 0, reason="no cable pixels left after removing the connectors")

    hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    voters = keep & (sat >= model.min_saturation) & (val >= model.min_value)
    n_sat = int(voters.sum())
    if n_sat < model.min_pixels:
        return ColorReading(None, 0.0, n_pixels, n_sat,
                            reason=f"only {n_sat} clearly coloured pixels (need {model.min_pixels})")

    counts = {}
    for name, ranges in model.ranges.items():
        m = np.zeros_like(voters)
        for lo, hi in ranges:
            m |= (hue >= lo) & (hue <= hi)
        counts[name] = int((m & voters).sum())
    shares = {k: v / n_sat for k, v in counts.items()}
    ranked = sorted(shares, key=lambda k: (-shares[k], k))  # name breaks exact ties deterministically
    best = ranked[0]
    runner_up = shares[ranked[1]] if len(ranked) > 1 else 0.0
    if shares[best] < model.min_share or shares[best] - runner_up < model.min_margin:
        return ColorReading(None, shares[best], n_pixels, n_sat, shares,
                            reason=f"no colour wins clearly (best: {best} {shares[best]:.0%}, "
                                   f"next {runner_up:.0%})")
    return ColorReading(best, shares[best], n_pixels, n_sat, shares)
