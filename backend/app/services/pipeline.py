from app.core.image_io import LoadedImage, load_image
from app.schemas.inspection import Detection, ImageInfo, InspectionResult
from app.schemas.spec import ProductSpec
from app.services.verification.count_rules import verify_counts


def run_inspection(image_path, spec: ProductSpec, detections: list[Detection],
                   detector_name: str) -> tuple[InspectionResult, LoadedImage]:
    img = load_image(image_path)
    for d in detections:
        b = d.bbox
        if b.x1 < 0 or b.y1 < 0 or b.x2 > img.width or b.y2 > img.height:
            raise ValueError(
                f"detection '{d.class_name}' bbox is outside the image ({img.width}x{img.height})"
            )
    status, findings = verify_counts(detections, spec)
    result = InspectionResult(
        product=spec.product,
        spec_version=spec.version,
        status=status,
        image=ImageInfo(name=img.name, width=img.width, height=img.height),
        detections=detections,
        findings=findings,
        detector=detector_name,
    )
    return result, img