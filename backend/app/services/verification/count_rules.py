from collections import Counter

from app.schemas.inspection import Detection, Finding, Status
from app.schemas.spec import ProductSpec


def verify_counts(detections: list[Detection], spec: ProductSpec) -> tuple[Status, list[Finding]]:
    """Minimal ST-01 rule: compare per-class counts with the spec.

    Detections are split into bands by confidence:
      >= accept_conf            -> confident
      [review_conf, accept)     -> uncertain
      < review_conf             -> ignored as noise
    A FAIL means "not enough evidence the part is there", which is only
    trustworthy once detector recall has been measured (ST-03).
    """
    t = spec.thresholds
    confident: Counter = Counter()
    uncertain: Counter = Counter()
    for d in detections:
        if d.confidence >= t.accept_conf:
            confident[d.class_name] += 1
        elif d.confidence >= t.review_conf:
            uncertain[d.class_name] += 1

    findings: list[Finding] = []
    for cls, expected in spec.required_counts.items():
        c, u = confident[cls], uncertain[cls]

        def add(code, status, msg):
            findings.append(Finding(code=code, status=status, class_name=cls,
                                    expected=expected, confident=c, uncertain=u, message=msg))

        if c > expected:
            add("EXTRA", Status.FAIL,
                f"{cls}: expected {expected}, found {c} confident (extra part?)")
        elif c == expected and u == 0:
            continue
        elif c == expected:
            add("UNCERTAIN_EXTRA", Status.REVIEW,
                f"{cls}: expected {expected}; {u} extra low-confidence detection(s) - needs human check")
        elif c + u >= expected:
            add("UNCERTAIN_PART", Status.REVIEW,
                f"{cls}: expected {expected}; only {c} confident, {u} low-confidence - needs human check")
        else:
            add("MISSING", Status.FAIL,
                f"{cls}: expected {expected}, found {c} confident + {u} uncertain")

    for cls in sorted(set(confident) - set(spec.required_counts)):
        findings.append(Finding(code="UNEXPECTED_CLASS", status=Status.REVIEW, class_name=cls,
                                confident=confident[cls],
                                message=f"{cls}: not in spec, {confident[cls]} confident detection(s)"))

    if any(f.status == Status.FAIL for f in findings):
        status = Status.FAIL
    elif any(f.status == Status.REVIEW for f in findings):
        status = Status.REVIEW
    else:
        status = Status.PASS
    return status, findings