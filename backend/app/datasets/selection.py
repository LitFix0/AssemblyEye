import random

from .loco import Sample


class LeakageError(ValueError):
    """An ineligible (test or anomalous) image was selected for annotation."""


def assert_no_leakage(chosen) -> None:
    for s, _role in chosen:
        if s.split == "test" or not s.is_normal:
            raise LeakageError(f"{s.rel_path} must not be annotated for training")


def _shuffled(samples: list[Sample], split: str, seed: int) -> list[Sample]:
    pool = sorted((s for s in samples if s.split == split and s.is_normal),
                  key=lambda s: s.rel_path)
    random.Random(f"{seed}:{split}").shuffle(pool)
    return pool


def select_for_annotation(samples, n_train_pool: int, n_holdout: int, seed: int = 42):
    """Return [(Sample, role)]. Smaller selections are prefixes of larger ones (same seed)."""
    train = _shuffled(samples, "train", seed)
    hold = _shuffled(samples, "validation", seed)
    if n_train_pool > len(train) or n_holdout > len(hold):
        raise ValueError(f"requested {n_train_pool}+{n_holdout} images but only "
                         f"{len(train)} train and {len(hold)} validation normals exist")
    chosen = ([(s, "train_pool") for s in train[:n_train_pool]]
              + [(s, "holdout") for s in hold[:n_holdout]])
    assert_no_leakage(chosen)
    return chosen