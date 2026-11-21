"""Expected-vs-detected table for display. Same confidence bands (band_of) and the same duplicate /
competing-class clean-up (resolve_all) as the verification rules, so the table counts exactly what the
verdict counted."""
from app.schemas.inspection import ComponentCount, Detection
from app.schemas.spec import ProductSpec
from app.services.verification.count_rules import band_of, resolve_all


def component_table(detections: list[Detection], spec: ProductSpec) -> list[ComponentCount]:
    detections, _ = resolve_all(detections, spec)

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