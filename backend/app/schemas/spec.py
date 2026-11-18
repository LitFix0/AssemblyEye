from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Thresholds(BaseModel):
    accept_conf: float
    review_conf: float

    @model_validator(mode="after")
    def _check(self):
        if not (0 < self.review_conf < self.accept_conf <= 1):
            raise ValueError("need 0 < review_conf < accept_conf <= 1")
        return self


class ColorModel(BaseModel):
    """How a cable colour is read from pixels. Hue is OpenCV hue (0-179), inclusive ranges.
    All numbers are placeholders until checked with scripts/check_splicing_rules.py."""
    ranges: dict[str, list[tuple[int, int]]]
    min_saturation: int = Field(80, ge=0, le=255)  # ignore grey/white/black pixels
    min_value: int = Field(50, ge=0, le=255)
    min_pixels: int = Field(40, ge=1)  # saturated pixels needed before a colour is trusted
    min_share: float = Field(0.5, gt=0, le=1)  # winning colour's share of the saturated pixels
    min_margin: float = Field(0.2, ge=0, le=1)  # winner must beat the runner-up by this much (ties are unknown)
    confident_share: float = Field(0.7, gt=0, le=1)  # share needed for a wrong colour to be FAIL, not REVIEW

    @field_validator("ranges")
    @classmethod
    def _ranges(cls, v):
        if not v:
            raise ValueError("colour ranges must not be empty")
        for name, rs in v.items():
            if not rs:
                raise ValueError(f"colour {name!r} has no hue ranges")
            for lo, hi in rs:
                if not (0 <= lo <= hi <= 179):
                    raise ValueError(f"colour {name!r}: hue range ({lo}, {hi}) must satisfy 0 <= lo <= hi <= 179")
        return v

    @model_validator(mode="after")
    def _order(self):
        if self.confident_share < self.min_share:
            raise ValueError("confident_share must be >= min_share")
        return self


class SplicingRules(BaseModel):
    """Relational rules for splicing_connectors: two connectors of the same type joined by one cable
    whose colour matches the connector type."""
    connector_classes: list[str]
    cable_class: str = "cable"
    n_connectors: Literal[2] = 2
    cable_colors: dict[str, str]  # connector class -> expected cable colour name
    end_tolerance: float = Field(0.02, gt=0, le=0.5)  # allowed cable-to-connector gap, fraction of image diagonal
    # Shortest allowed cable along the axis between the connectors, fraction of image diagonal. None = off.
    min_cable_length: float | None = Field(None, gt=0, le=1)
    color: ColorModel

    @model_validator(mode="after")
    def _check(self):
        if len(self.connector_classes) < 1 or len(set(self.connector_classes)) != len(self.connector_classes):
            raise ValueError("connector_classes must be non-empty and unique")
        if self.cable_class in self.connector_classes:
            raise ValueError("cable_class must not also be a connector class")
        if set(self.cable_colors) != set(self.connector_classes):
            raise ValueError("cable_colors needs exactly one entry per connector class")
        unknown = set(self.cable_colors.values()) - set(self.color.ranges)
        if unknown:
            raise ValueError(f"cable_colors uses colours with no hue range: {sorted(unknown)}")
        return self


class ProductSpec(BaseModel):
    product: str
    version: str
    description: str = ""
    thresholds: Thresholds
    required_counts: dict[str, int] = Field(default_factory=dict)
    # Per-class override of the global thresholds (e.g. a class the detector is systematically less sure of).
    class_thresholds: dict[str, Thresholds] = Field(default_factory=dict)
    # Groups of classes the detector can confuse on the same object (e.g. long vs short screw).
    # A confusion is a REVIEW, never a silent PASS and never an automatic FAIL.
    competing_classes: list[list[str]] = Field(default_factory=list)
    competing_iou: float = Field(0.7, gt=0, le=1)
    # Same-class boxes overlapping at least this much are one object seen twice: counted once, REVIEW.
    # None = off (the detector's own NMS is then the only duplicate handling).
    duplicate_iou: float | None = Field(None, gt=0, le=1)
    # A below-accept box overlapping a STRONGER same-class box by at least this IoU is the same object
    # detected again: dropped without a finding. None = off.
    weak_duplicate_iou: float | None = Field(None, gt=0, le=1)
    splicing: SplicingRules | None = None

    @field_validator("required_counts")
    @classmethod
    def _counts(cls, v):
        if any(n < 0 for n in v.values()):
            raise ValueError("counts must be >= 0")
        return v

    @model_validator(mode="after")
    def _check(self):
        if not self.required_counts and self.splicing is None:
            raise ValueError("required_counts must not be empty")
        for group in self.competing_classes:
            if len(group) < 2 or len(set(group)) != len(group):
                raise ValueError(f"competing group needs 2+ distinct classes: {group}")
            if self.splicing is None:
                missing = [c for c in group if c not in self.required_counts]
                if missing:
                    raise ValueError(f"competing group {group} has classes not in required_counts: {missing}")
        return self

    def thresholds_for(self, class_name: str) -> Thresholds:
        return self.class_thresholds.get(class_name, self.thresholds)

    @property
    def min_review_conf(self) -> float:
        """Lowest review threshold in the spec; the detector must report at least this low."""
        return min([self.thresholds.review_conf] + [t.review_conf for t in self.class_thresholds.values()])


def load_spec(path) -> ProductSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ProductSpec.model_validate(data)
