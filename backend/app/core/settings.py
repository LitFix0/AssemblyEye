"""Serving settings. Paths come from configs/serving.yaml (relative to the repo root) and can be
overridden with environment variables, so nothing is machine-specific.

  VAV_CONFIG    path to a serving yaml        (default: <repo>/configs/serving.yaml)
  VAV_DATA_DIR  where evidence images are kept (default: <repo>/data/inspections)
  VAV_DB        SQLite file for inspection history (default: <repo>/data/vav.db)
"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]


class ProductServing(BaseModel):
    spec: str
    weights: str
    imgsz: int = Field(640, ge=32, le=4096)
    cross_class_suppression: bool = False


class ServingConfig(BaseModel):
    products: dict[str, ProductServing]


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    db_path: Path
    serving: ServingConfig

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.root / path


def load_settings(config_path=None, data_dir=None, db_path=None, root: Path = REPO_ROOT) -> Settings:
    cfg = Path(config_path or os.environ.get("VAV_CONFIG") or root / "configs" / "serving.yaml")
    serving = ServingConfig.model_validate(yaml.safe_load(cfg.read_text(encoding="utf-8")))
    data = Path(data_dir or os.environ.get("VAV_DATA_DIR") or root / "data" / "inspections")
    db = Path(db_path or os.environ.get("VAV_DB") or root / "data" / "vav.db")
    return Settings(root=root, data_dir=data, db_path=db, serving=serving)