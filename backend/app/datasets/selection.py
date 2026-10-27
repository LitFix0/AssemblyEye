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


def select_for_annotation(samples, n_train_pool: int, n_holdout: int, seed: int = 42,
                          reserve_dev: int = 0):
    """Return [(Sample, role)]. Smaller selections are prefixes of larger ones (same seed).
    The last `reserve_dev` train images of the shuffle are never selected (see select_normal_dev)."""
    train = _shuffled(samples, "train", seed)
    hold = _shuffled(samples, "validation", seed)
    available = len(train) - reserve_dev
    if n_train_pool > available or n_holdout > len(hold):
        raise ValueError(f"requested {n_train_pool}+{n_holdout} images but only {available} train "
                         f"(after reserving {reserve_dev} for the dev set) and {len(hold)} "
                         "validation normals exist")
    chosen = ([(s, "train_pool") for s in train[:n_train_pool]]
              + [(s, "holdout") for s in hold[:n_holdout]])
    assert_no_leakage(chosen)
    return chosen


def select_normal_dev(samples, n: int, seed: int = 42):
    """Normal-only dev set: the LAST n images of the seeded train/good shuffle.
    Annotation batches take the front of the same shuffle, so the two never overlap
    when select_for_annotation is called with reserve_dev >= n."""
    train = _shuffled(samples, "train", seed)
    if n > len(train):
        raise ValueError(f"requested {n} dev images but only {len(train)} train normals exist")
    chosen = train[len(train) - n:]
    assert_no_leakage([(s, "normal_dev") for s in chosen])
    return chosen