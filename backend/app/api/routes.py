from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse

from app.api.errors import ApiError
from app.core.image_io import MAX_BYTES, ImageValidationError
from app.schemas.api import ErrorResponse, HealthResponse, InspectionResponse, ProductInfo
from app.services.inspection_service import ID_RE, inspect_bytes

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
                             request.app.state.settings.data_dir)
    except ImageValidationError as e:
        raise ApiError(422, "invalid_image", str(e))


def _folder(request: Request, iid: str):
    folder = request.app.state.settings.data_dir / iid
    if not ID_RE.match(iid) or not folder.is_dir():
        raise ApiError(404, "not_found", "no such inspection")
    return folder


@router.get("/inspections/{iid}", response_model=InspectionResponse, responses=ERRORS)
def get_inspection(request: Request, iid: str):
    return InspectionResponse.model_validate_json((_folder(request, iid) / "result.json").read_text("utf-8"))


@router.get("/inspections/{iid}/{kind}", responses=ERRORS)
def get_image(request: Request, iid: str, kind: str):
    folder = _folder(request, iid)
    if kind not in ("original", "annotated"):
        raise ApiError(404, "not_found", "kind must be 'original' or 'annotated'")
    matches = sorted(folder.glob(f"{kind}.*"))
    if not matches:
        raise ApiError(404, "not_found", "evidence file missing")
    return FileResponse(matches[0])