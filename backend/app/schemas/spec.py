from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator


class Thresholds(BaseModel):
    accept_conf: float
    review_conf: float

    @model_validator(mode="after")
    def _check(self):
        if not (0 < self.review_conf < self.accept_conf <= 1):
            raise ValueError("need 0 < review_conf < accept_conf <= 1")
        return self


class ProductSpec(BaseModel):
    product: str
    version: str
    description: str = ""
    thresholds: Thresholds
    required_counts: dict[str, int]

    @field_validator("required_counts")
    @classmethod
    def _counts(cls, v):
        if not v:
            raise ValueError("required_counts must not be empty")
        if any(n < 0 for n in v.values()):
            raise ValueError("counts must be >= 0")
        return v


def load_spec(path) -> ProductSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ProductSpec.model_validate(data)