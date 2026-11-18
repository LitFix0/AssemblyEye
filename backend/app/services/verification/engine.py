"""Single entry point for the verification rules: picks the rule set that matches the product spec."""
import numpy as np

from app.schemas.inspection import Detection, Finding, Status
from app.schemas.spec import ProductSpec
from app.services.verification.count_rules import verify_counts
from app.services.verification.splicing import verify_splicing


def verify_product(detections: list[Detection], spec: ProductSpec,
                   image_bgr: np.ndarray | None = None) -> tuple[Status, list[Finding]]:
    """Pure function of its inputs: same detections + spec + image -> same decision.

    Count-only products (screw_bag) never look at pixels. Splicing needs the image because the cable
    colour is read from pixels; passing none is an error, not a silent skip.
    """
    if spec.splicing is not None:
        if image_bgr is None:
            raise ValueError("splicing verification needs the image (cable colour is read from pixels)")
        return verify_splicing(detections, spec, image_bgr)
    return verify_counts(detections, spec)
