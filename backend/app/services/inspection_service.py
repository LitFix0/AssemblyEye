"""One inspection: store evidence, validate, detect, verify, annotate. No HTTP in here."""
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.image_io import ALLOWED_EXT, ImageValidationError, load_image, save_image
from app.evaluation.counts import suppress_cross_class
from app.schemas.api import Evidence, InspectionResponse
from app.services.pipeline import run_inspection
from app.services.registry import DetectorRegistry
from app.services.visualization.draw import draw_result

ID_RE = re.compile(r"^[0-9a-f]{32}$")


def safe_name(filename: str | None) -> str:
    name = Path((filename or "").replace("\\", "/")).name
    return re.sub(r"[^\w.\- ]", "_", name)[:100] or "upload"


def inspect_bytes(data: bytes, filename: str | None, product: str, registry: DetectorRegistry,
                  data_dir: Path) -> InspectionResponse:
    t0 = time.perf_counter()
    entry = registry.get(product)
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ImageValidationError(f"unsupported file type: {ext or '(none)'}")
    detector = registry.detector_for(entry)

    iid = uuid4().hex
    folder = data_dir / iid
    folder.mkdir(parents=True)
    try:
        original = folder / f"original{ext}"
        original.write_bytes(data)
        img = load_image(original)  # validates size, decodability, dimensions
        t1 = time.perf_counter()
        with entry.run_lock:
            dets = detector.detect(img.bgr)
        t2 = time.perf_counter()
        if entry.suppress:
            dets = suppress_cross_class(dets)
        result, _ = run_inspection(original, entry.spec, dets, detector.name)
        result.image.name = safe_name(filename)
        save_image(folder / "annotated.png", draw_result(img.bgr, result, entry.spec))
        response = InspectionResponse(
            inspection_id=iid, created_at=datetime.now(timezone.utc), result=result,
            evidence=Evidence(original_url=f"/inspections/{iid}/original",
                              annotated_url=f"/inspections/{iid}/annotated"),
            timings_ms={"load": round((t1 - t0) * 1000, 1), "detect": round((t2 - t1) * 1000, 1),
                        "total": round((time.perf_counter() - t0) * 1000, 1)})
        (folder / "result.json").write_text(response.model_dump_json(indent=2), encoding="utf-8")
        return response
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)  # no evidence folder for a request that failed
        raise