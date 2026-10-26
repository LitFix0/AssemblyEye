import json
from pathlib import Path

from app.schemas.inspection import Detection

DETECTOR_NAME = "fixture (hand-written detections, NOT a trained model)"


def load_fixture(path) -> list[Detection]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Detection.model_validate(d) for d in raw["detections"]]