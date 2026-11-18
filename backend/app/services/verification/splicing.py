"""Relational rules for splicing_connectors.

A correct assembly is: exactly two connectors of the SAME type (same clamp count), exactly one cable
whose two ends reach the two connectors, and a cable colour that matches the connector type
(yellow = 2, blue = 3, red = 5 in the spec). Each rule produces a finding with the numbers behind it.

Order of work
  1. Band every detection by confidence (per-class thresholds: cable is less confident than connectors).
  2. Connector classes that compete on one object -> REVIEW, keep the more confident reading.
  3. Count rules for connectors (need 2) and cable (need 1). Anything not exactly right stops the
     geometry checks for that image: with a part missing or doubtful there is nothing sound to measure.
  4. The two connectors must be the same type.
  5. Cable ends: the gap between the cable box and each connector box must be small, and the cable
     must not stick out past the far side of a connector. Works along whichever axis separates the
     connectors (left/right in the dataset images).
  6. Cable colour from pixels must match the connector type; unreadable colour is REVIEW.
FAIL = a confirmed requirement is violated; REVIEW = uncertain or not measurable; PASS only if nothing fired.
"""
import math

from app.schemas.inspection import BBox, Detection, Finding, Status
from app.schemas.spec import ProductSpec
from app.services.verification.color import read_cable_color
from app.services.verification.count_rules import (band_of, count_finding, resolve_competing,
                                                   resolve_duplicates, status_of)


def box_gap(a: BBox, b: BBox) -> float:
    """Shortest distance between two boxes in px; 0 if they touch or overlap."""
    dx = max(0.0, max(a.x1, b.x1) - min(a.x2, b.x2))
    dy = max(0.0, max(a.y1, b.y1) - min(a.y2, b.y2))
    return math.hypot(dx, dy)


def _box(b: BBox) -> list[float]:
    return [round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1)]


def _centre(b: BBox) -> tuple[float, float]:
    return (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2


def _extent(b: BBox, axis: int) -> tuple[float, float]:
    return (b.x1, b.x2) if axis == 0 else (b.y1, b.y2)


def verify_splicing(detections: list[Detection], spec: ProductSpec, image_bgr) -> tuple[Status, list[Finding]]:
    rules = spec.splicing
    if rules is None:
        raise ValueError("spec has no splicing rules")
    h, w = image_bgr.shape[:2]
    findings: list[Finding] = []
    known = set(rules.connector_classes) | {rules.cable_class}

    dets, dup = resolve_duplicates(detections, spec)
    dets, comp = resolve_competing(dets, list(rules.connector_classes), spec, spec.competing_iou)
    findings += dup + comp

    conf_conn = [d for d in dets if d.class_name in rules.connector_classes and band_of(d, spec) == "confident"]
    unc_conn = [d for d in dets if d.class_name in rules.connector_classes and band_of(d, spec) == "uncertain"]
    conf_cab = [d for d in dets if d.class_name == rules.cable_class and band_of(d, spec) == "confident"]
    unc_cab = [d for d in dets if d.class_name == rules.cable_class and band_of(d, spec) == "uncertain"]

    for label, expected, c, u in (("connector", rules.n_connectors, len(conf_conn), len(unc_conn)),
                                  (rules.cable_class, 1, len(conf_cab), len(unc_cab))):
        res = count_finding(label, expected, c, u)
        if res:
            code, status, msg = res
            findings.append(Finding(code=code, status=status, class_name=label, expected=expected,
                                    confident=c, uncertain=u, message=msg))

    other = sorted({d.class_name for d in dets if d.class_name not in known and band_of(d, spec) == "confident"})
    for cls in other:
        n = sum(1 for d in dets if d.class_name == cls and band_of(d, spec) == "confident")
        findings.append(Finding(code="UNEXPECTED_CLASS", status=Status.REVIEW, class_name=cls, confident=n,
                                message=f"{cls}: not in spec, {n} confident detection(s)"))

    # Geometry and colour need exactly two confident connectors and exactly one confident cable.
    if len(conf_conn) == rules.n_connectors and len(conf_cab) == 1:
        a, b = conf_conn
        types = sorted({a.class_name, b.class_name})
        if len(types) > 1:
            findings.append(Finding(
                code="CONNECTOR_TYPE_MISMATCH", status=Status.FAIL, class_name="connector",
                message=f"The two connectors differ ({types[0]} and {types[1]}); a pair must have the same clamp count.",
                evidence={"connectors": [{"class": d.class_name, "confidence": round(d.confidence, 3),
                                          "box": _box(d.bbox)} for d in (a, b)]}))
        findings += _check_cable_ends(a, b, conf_cab[0], rules, w, h)
        if len(types) == 1:
            findings += _check_cable_color(a, b, conf_cab[0], image_bgr, rules, types[0])
    return status_of(findings), findings


def _check_cable_ends(a: Detection, b: Detection, cable: Detection, rules, w: int, h: int):
    ca, cb = _centre(a.bbox), _centre(b.bbox)
    axis = 0 if abs(ca[0] - cb[0]) >= abs(ca[1] - cb[1]) else 1
    first, second = (a, b) if ca[axis] <= cb[axis] else (b, a)
    names = ("left", "right") if axis == 0 else ("upper", "lower")
    diag = math.hypot(w, h)
    tol = rules.end_tolerance * diag
    c_lo, c_hi = _extent(cable.bbox, axis)
    out: list[Finding] = []

    if rules.min_cable_length is not None:
        length, minimum = c_hi - c_lo, rules.min_cable_length * diag
        if length < minimum:
            out.append(Finding(
                code="CABLE_TOO_SHORT", status=Status.FAIL, class_name=cable.class_name,
                message=f"Cable is too short: {length:.0f} px between its ends, minimum {minimum:.0f} px.",
                evidence={"length_px": round(length, 1), "minimum_px": round(minimum, 1),
                          "cable_box": _box(cable.bbox)}))

    for side, conn in zip(names, (first, second)):
        gap = box_gap(cable.bbox, conn.bbox)
        ev = {"side": side, "gap_px": round(gap, 1), "tolerance_px": round(tol, 1),
              "cable_box": _box(cable.bbox), "connector_box": _box(conn.bbox)}
        if gap > tol:
            out.append(Finding(
                code="CABLE_NOT_CONNECTED", status=Status.FAIL, class_name=cable.class_name,
                message=(f"Cable does not reach the {side} connector: gap {gap:.0f} px "
                         f"(allowed {tol:.0f} px)."),
                evidence=ev))

    lo_conn, hi_conn = _extent(first.bbox, axis), _extent(second.bbox, axis)
    for side, over, conn in ((names[0], lo_conn[0] - c_lo, first), (names[1], c_hi - hi_conn[1], second)):
        if over > tol:
            out.append(Finding(
                code="CABLE_MISPLACED", status=Status.FAIL, class_name=cable.class_name,
                message=(f"Cable sticks out {over:.0f} px past the far side of the {side} connector "
                         f"(allowed {tol:.0f} px)."),
                evidence={"side": side, "overshoot_px": round(over, 1), "tolerance_px": round(tol, 1),
                          "cable_box": _box(cable.bbox), "connector_box": _box(conn.bbox)}))
    return out


def _check_cable_color(a, b, cable, image_bgr, rules, connector_class):
    expected = rules.cable_colors[connector_class]
    reading = read_cable_color(image_bgr, cable.bbox, [a.bbox, b.bbox], rules.color)
    ev = {"expected": expected, "observed": reading.name, "share": round(reading.share, 3),
          "shares": {k: round(v, 3) for k, v in reading.shares.items()},
          "pixels_examined": reading.n_pixels, "pixels_coloured": reading.n_saturated,
          "cable_box": _box(cable.bbox)}
    if reading.name is None:
        return [Finding(code="CABLE_COLOR_UNKNOWN", status=Status.REVIEW, class_name=cable.class_name,
                        message=f"Cable colour could not be read ({reading.reason}); {connector_class} needs a "
                                f"{expected} cable. Check it by eye.",
                        evidence={**ev, "reason": reading.reason})]
    if reading.name == expected:
        return []
    sure = reading.share >= rules.color.confident_share
    return [Finding(
        code="CABLE_COLOR_MISMATCH", status=Status.FAIL if sure else Status.REVIEW, class_name=cable.class_name,
        message=(f"Cable looks {reading.name} ({reading.share:.0%} of coloured pixels) but {connector_class} "
                 f"needs {expected}." + ("" if sure else " Colour reading is not clear-cut; check it by eye.")),
        evidence=ev)]