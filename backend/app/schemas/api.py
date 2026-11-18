from datetime import datetime

from pydantic import BaseModel

from app.schemas.inspection import InspectionResult


class Evidence(BaseModel):
    original_url: str
    annotated_url: str


class InspectionResponse(BaseModel):
    inspection_id: str
    created_at: datetime
    result: InspectionResult
    evidence: Evidence
    timings_ms: dict[str, float]


class ProductInfo(BaseModel):
    product: str
    spec_version: str
    description: str
    weights_present: bool
    detector_loaded: bool


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded" (a product's weights file is missing)
    products: list[ProductInfo]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody