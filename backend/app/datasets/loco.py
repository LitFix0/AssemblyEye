"""Read-only adapter for the MVTec LOCO AD directory layout."""
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "validation", "test")
LABELS = ("good", "logical_anomalies", "structural_anomalies")


class DatasetLayoutError(RuntimeError):
    """The directory does not look like MVTec LOCO AD."""


@dataclass(frozen=True)
class Sample:
    category: str
    split: str  # train | validation | test
    label: str  # good | logical_anomalies | structural_anomalies
    image_path: Path
    rel_path: str  # posix path relative to the category folder
    mask_paths: tuple[Path, ...] = ()

    @property
    def is_normal(self) -> bool:
        return self.label == "good"


def load_defect_names(category_dir) -> dict[int, str]:
    cfg = json.loads((Path(category_dir) / "defects_config.json").read_text(encoding="utf-8"))
    return {int(d["pixel_value"]): d["defect_name"] for d in cfg}


def index_category(root, category: str) -> list[Sample]:
    base = Path(root) / category
    if not base.is_dir():
        raise DatasetLayoutError(f"category folder not found: {base}")
    samples: list[Sample] = []
    for split in SPLITS:
        for label in LABELS:
            folder = base / split / label
            if not folder.is_dir():
                continue
            if split != "test" and label != "good":
                raise DatasetLayoutError(f"unexpected anomalous images in {folder}")
            for img in sorted(folder.glob("*.png")):
                masks: tuple[Path, ...] = ()
                if label != "good":
                    masks = tuple(sorted((base / "ground_truth" / label / img.stem).glob("*.png")))
                    if not masks:
                        raise DatasetLayoutError(f"no ground-truth mask for {img}")
                samples.append(Sample(category, split, label, img,
                                      img.relative_to(base).as_posix(), masks))
    if not any(s.split == "train" for s in samples):
        raise DatasetLayoutError(f"no training images under {base / 'train'}")
    return samples


def summarize(samples) -> dict[str, int]:
    return dict(sorted(Counter(f"{s.split}/{s.label}" for s in samples).items()))


def defect_values(sample: Sample) -> list[int]:
    """Distinct non-zero pixel values over all masks of an anomalous sample."""
    vals: set[int] = set()
    for m in sample.mask_paths:
        with Image.open(m) as im:
            vals |= {int(v) for v in np.unique(np.array(im))}
    vals.discard(0)
    return sorted(vals)