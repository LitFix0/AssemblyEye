from collections import Counter

from app.evaluation.counts import iou
from app.schemas.inspection import Detection, Finding, Status
from app.schemas.spec import ProductSpec


def band_of(d: Detection, spec: ProductSpec) -> str:
    """'confident' | 'uncertain' | 'ignored', using the class's own thresholds if it has any."""
    t = spec.thresholds_for(d.class_name)
    if d.confidence >= t.accept_conf:
        return "confident"
    if d.confidence >= t.review_conf:
        return "uncertain"
    return "ignored"


def count_finding(label: str, expected: int, c: int, u: int):
    """The count rule for one class or group. c = confident count, u = uncertain count.
    Returns (code, status, message) or None when the count is exactly right."""
    if c > expected:
        return "EXTRA", Status.FAIL, f"{label}: expected {expected}, found {c} confident (extra part?)"
    if c == expected and u == 0:
        return None
    if c == expected:
        return ("UNCERTAIN_EXTRA", Status.REVIEW,
                f"{label}: expected {expected}; {u} extra low-confidence detection(s) - needs human check")
    if c + u >= expected:
        return ("UNCERTAIN_PART", Status.REVIEW,
                f"{label}: expected {expected}; only {c} confident, {u} low-confidence - needs human check")
    return "MISSING", Status.FAIL, f"{label}: expected {expected}, found {c} confident + {u} uncertain"


def status_of(findings: list[Finding]) -> Status:
    if any(f.status == Status.FAIL for f in findings):
        return Status.FAIL
    if any(f.status == Status.REVIEW for f in findings):
        return Status.REVIEW
    return Status.PASS


def _ev(d: Detection) -> dict:
    b = d.bbox
    return {"class": d.class_name, "confidence": round(d.confidence, 3),
            "box": [round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1)]}


def resolve_duplicates(detections: list[Detection], spec: ProductSpec) -> tuple[list[Detection], list[Finding]]:
    """Same class, heavily overlapping boxes = one object detected twice. Keep the more confident box
    so the part is counted once, and raise a REVIEW finding (a detector artefact worth a human glance;
    it must not become a false EXTRA FAIL, and it must not pass silently either)."""
    if spec.duplicate_iou is None and spec.weak_duplicate_iou is None:
        return list(detections), []
    kept: list[Detection] = []
    dropped: list[Detection] = []
    findings: list[Finding] = []
    for d in sorted((x for x in detections if band_of(x, spec) != "ignored"), key=lambda x: -x.confidence):
        twin = None if spec.duplicate_iou is None else next(
            (k for k in kept if k.class_name == d.class_name and iou(k.bbox, d.bbox) >= spec.duplicate_iou), None)
        if twin is None:
            kept.append(d)
            continue
        dropped.append(d)
        findings.append(Finding(
            code="DUPLICATE_DETECTION", status=Status.REVIEW, class_name=d.class_name,
            message=(f"Two overlapping {d.class_name} boxes ({twin.confidence:.2f} and {d.confidence:.2f}) "
                     f"look like one part detected twice; counted once. Check it by eye."),
            evidence={"kept": _ev(twin), "dropped": _ev(d), "iou": round(iou(twin.bbox, d.bbox), 3)}))
    if spec.weak_duplicate_iou is not None:
        # A below-accept box sitting on a stronger box of its class is that object again, not a second
        # object. Dropped without a finding: the object is still judged through the stronger box.
        for d in [k for k in kept if band_of(k, spec) == "uncertain"]:
            if any(k is not d and k.class_name == d.class_name and k.confidence > d.confidence
                   and iou(k.bbox, d.bbox) >= spec.weak_duplicate_iou for k in kept):
                dropped.append(d)
    drop_ids = {id(d) for d in dropped}
    return [d for d in detections if id(d) not in drop_ids], findings


def resolve_competing(detections: list[Detection], group: list[str], spec: ProductSpec,
                      iou_thr: float) -> tuple[list[Detection], list[Finding]]:
    """One physical object read as two classes of the same competing group (e.g. screw_long AND
    screw_short on the same box). Keep the more confident reading for counting, drop the other, and
    raise a REVIEW finding: the detector is not sure which class this part is, so a person should look.
    Ignored (below review_conf) detections are never involved."""
    in_group = sorted((d for d in detections if d.class_name in group and band_of(d, spec) != "ignored"),
                      key=lambda d: -d.confidence)
    kept: list[Detection] = []
    dropped: list[Detection] = []
    findings: list[Finding] = []
    for d in in_group:
        rival = next((k for k in kept if k.class_name != d.class_name and iou(k.bbox, d.bbox) >= iou_thr), None)
        if rival is None:
            kept.append(d)
            continue
        dropped.append(d)
        findings.append(Finding(
            code="COMPETING_CLASSES", status=Status.REVIEW, class_name=rival.class_name,
            message=(f"One object is read as both {rival.class_name} ({rival.confidence:.2f}) and "
                     f"{d.class_name} ({d.confidence:.2f}); counted as {rival.class_name}. Check it by eye."),
            evidence={"kept": _ev(rival), "dropped": _ev(d), "iou": round(iou(rival.bbox, d.bbox), 3)}))
    drop_ids = {id(d) for d in dropped}
    return [d for d in detections if id(d) not in drop_ids], findings


def _merge_class_splits(findings: list[Finding], spec: ProductSpec, confident: Counter,
                        uncertain: Counter) -> list[Finding]:
    """A competing group can have the right number of parts and still not match class by class:
      (a) confident parts alone add up to the right total but split wrongly (2 x screw_long, 0 x
          screw_short), or
      (b) too few confident parts, but confident + low-confidence ones add up to the right total
          (the missing screw may be sitting there as a low-confidence reading of the other class).
    Neither is established as a missing or extra part. It is a length defect or a long/short
    misclassification and the system cannot tell which, so the FAIL becomes a REVIEW."""
    out = list(findings)
    for group in spec.competing_classes:
        expected = {c: spec.required_counts[c] for c in group}
        found = {c: confident[c] for c in group}
        low = {c: uncertain[c] for c in group}
        exp_total, conf_total, low_total = sum(expected.values()), sum(found.values()), sum(low.values())
        split = conf_total == exp_total and found != expected  # (a)
        short_but_covered = conf_total < exp_total <= conf_total + low_total  # (b)
        if not (split or short_but_covered):
            continue
        replace = [f for f in out if f.class_name in group and f.code in ("EXTRA", "MISSING")]
        if not replace:
            continue
        out = [f for f in out if f not in replace]
        detail = "; ".join(f"{c} {found[c]}" + (f" + {low[c]} low-confidence" if low[c] else "")
                           + f" (expected {expected[c]})" for c in group)
        lead = (f"right total ({conf_total}) but wrong split" if split else
                f"{conf_total} confident, {low_total} low-confidence: enough candidates but the split is unclear")
        out.append(Finding(
            code="CLASS_SPLIT", status=Status.REVIEW, class_name="/".join(group),
            expected=exp_total, confident=conf_total, uncertain=low_total,
            message=(f"{'/'.join(group)}: {lead}: {detail}. "
                     f"A part may be the wrong length or mis-classified. Check it by eye."),
            evidence={"expected": expected, "found": found, "low_confidence": low}))
    return out


def verify_counts(detections: list[Detection], spec: ProductSpec) -> tuple[Status, list[Finding]]:
    """Per-class count rules.

    Detections are split into bands by confidence (per class if the spec gives class_thresholds):
      >= accept_conf            -> confident
      [review_conf, accept)     -> uncertain
      < review_conf             -> ignored as noise
    A FAIL means "not enough evidence the part is there", which is only trustworthy once detector
    recall has been measured (ST-03).

    If the spec lists competing_classes (confusable classes), two extra rules apply: the same object
    read as two classes of a group, and a right-total-wrong-split inside a group, both give REVIEW.
    Specs without competing_classes behave exactly as the original ST-01 rule.
    """
    dets, findings = resolve_duplicates(detections, spec)
    for group in spec.competing_classes:
        dets, comp = resolve_competing(dets, group, spec, spec.competing_iou)
        findings += comp

    confident: Counter = Counter()
    uncertain: Counter = Counter()
    for d in dets:
        band = band_of(d, spec)
        if band == "confident":
            confident[d.class_name] += 1
        elif band == "uncertain":
            uncertain[d.class_name] += 1

    count_findings: list[Finding] = []
    for cls, expected in spec.required_counts.items():
        c, u = confident[cls], uncertain[cls]
        res = count_finding(cls, expected, c, u)
        if res:
            code, status, msg = res
            count_findings.append(Finding(code=code, status=status, class_name=cls,
                                          expected=expected, confident=c, uncertain=u, message=msg))
    findings += _merge_class_splits(count_findings, spec, confident, uncertain)

    for cls in sorted(set(confident) - set(spec.required_counts)):
        findings.append(Finding(code="UNEXPECTED_CLASS", status=Status.REVIEW, class_name=cls,
                                confident=confident[cls],
                                message=f"{cls}: not in spec, {confident[cls]} confident detection(s)"))
    return status_of(findings), findings
