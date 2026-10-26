from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_order(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
        return self


class Detection(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox


class Finding(BaseModel):
    code: str  # MISSING, EXTRA, UNCERTAIN_PART, UNCERTAIN_EXTRA, UNEXPECTED_CLASS
    status: Status  # FAIL or REVIEW: what this finding contributes
    class_name: str
    expected: int | None = None
    confident: int | None = None
    uncertain: int | None = None
    message: str


class ImageInfo(BaseModel):
    name: str
    width: int
    height: int


class InspectionResult(BaseModel):
    product: str
    spec_version: str
    status: Status
    image: ImageInfo
    detections: list[Detection]
    findings: list[Finding]
    detector: str