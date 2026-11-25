import logging
import shutil
from datetime import datetime

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from app.api.errors import ApiError
from app.core.image_io import MAX_BYTES, ImageValidationError
from app.schemas.api import (ErrorResponse, HealthResponse, InspectionList, InspectionResponse, ProductInfo)
from app.schemas.inspection import Status
from app.services.inspection_service import ID_RE, inspect_bytes

log = logging.getLogger("vav.api")
router = APIRouter()
ERRORS = {404: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
          503: {"model": ErrorResponse}}


def _products(request: Request) -> list[ProductInfo]:
    return [ProductInfo(product=e.name, spec_version=e.spec.version, description=e.spec.description.strip(),
                        weights_present=e.weights.is_file(), detector_loaded=e.detector is not None)
            for e in request.app.state.registry.entries.values()]


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    products = _products(request)
    return HealthResponse(status="ok" if all(p.weights_present for p in products) else "degraded",
                          products=products)


@router.get("/products", response_model=list[ProductInfo])
def products(request: Request):
    return _products(request)


@router.post("/inspect", response_model=InspectionResponse, responses=ERRORS)
def inspect(request: Request, product: str = Form(...), file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ApiError(413, "file_too_large", f"file larger than {MAX_BYTES} bytes")
    try:
        return inspect_bytes(data, file.filename, product, request.app.state.registry,
                             request.app.state.settings.data_dir, request.app.state.store)
    except ImageValidationError as e:
        raise ApiError(422, "invalid_image", str(e))


@router.get("/inspections", response_model=InspectionList, responses=ERRORS)
def list_inspections(request: Request, product: str | None = None, status: Status | None = None,
                     code: str | None = None, since: datetime | None = None, until: datetime | None = None,
                     limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    """Newest first. `since` is inclusive, `until` exclusive; naive datetimes are read as UTC."""
    return request.app.state.store.search(product=product, status=status.value if status else None, code=code,
                                          since=since, until=until, limit=limit, offset=offset)


def _valid_id(iid: str) -> str:
    if not ID_RE.match(iid):
        raise ApiError(404, "not_found", "no such inspection")
    return iid


@router.get("/inspections/{iid}", response_model=InspectionResponse, responses=ERRORS)
def get_inspection(request: Request, iid: str):
    found = request.app.state.store.get(_valid_id(iid))
    if found is None:
        raise ApiError(404, "not_found", "no such inspection")
    return found


@router.delete("/inspections/{iid}", status_code=204, responses=ERRORS)
def delete_inspection(request: Request, iid: str):
    """Removes the record and its images. There is no authentication in this prototype: do not expose it."""
    _valid_id(iid)
    folder = request.app.state.settings.data_dir / iid
    had_row = request.app.state.store.delete(iid)
    had_folder = folder.is_dir()
    if had_folder:
        shutil.rmtree(folder, ignore_errors=True)
        if folder.exists():
            log.error("record deleted but evidence folder could not be removed: %s", folder)
    if not (had_row or had_folder):
        raise ApiError(404, "not_found", "no such inspection")
    return Response(status_code=204)


@router.get("/inspections/{iid}/{kind}", responses=ERRORS)
def get_image(request: Request, iid: str, kind: str):
    folder = request.app.state.settings.data_dir / _valid_id(iid)
    if not folder.is_dir():
        raise ApiError(404, "not_found", "no such inspection")
    if kind not in ("original", "annotated"):
        raise ApiError(404, "not_found", "kind must be 'original' or 'annotated'")
    matches = sorted(folder.glob(f"{kind}.*"))
    if not matches:
        raise ApiError(404, "not_found", "evidence file missing")
    return FileResponse(matches[0])