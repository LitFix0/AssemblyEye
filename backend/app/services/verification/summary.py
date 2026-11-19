"""Expected-vs-detected table for display. Same confidence bands as the verification rules
(band_of), so the table and the findings cannot disagree about what counts as confident."""
from app.schemas.inspection import ComponentCount, Detection
from app.schemas.spec import ProductSpec
from app.services.verification.count_rules import band_of


def component_table(detections: list[Detection], spec: ProductSpec) -> list[ComponentCount]:
    def tally(classes: set[str]) -> tuple[int, int]:
        bands = [band_of(d, spec) for d in detections if d.class_name in classes]
        return bands.count("confident"), bands.count("uncertain")

    if spec.splicing is not None:  # all connector types count together; the type match is a separate rule
        rules = spec.splicing
        rows = [("connector", rules.n_connectors, set(rules.connector_classes)),
                (rules.cable_class, 1, {rules.cable_class})]
    else:
        rows = [(cls, n, {cls}) for cls, n in spec.required_counts.items()]
    return [ComponentCount(class_name=name, expected=n, confident=c, uncertain=u)
            for name, n, classes in rows for c, u in [tally(classes)]]