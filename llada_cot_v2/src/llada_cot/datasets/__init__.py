"""Dataset implementations."""
from dataclasses import dataclass
from typing import Any, Optional, List

from ..config import DatasetConfig, DatasetType


@dataclass
class DatasetExample:
    """A single example from a dataset."""
    idx: int
    question: str
    gold_answer: Any
    raw_data: dict
    numbers: Optional[List[int]] = None
    target: Optional[int] = None


def create_dataset(config: DatasetConfig):
    """Factory function to create and load a dataset."""
    if config.type == DatasetType.BIGGSM:
        from .biggsm import load_biggsm
        return load_biggsm(config.n_eval, config.seed)
    elif config.type == DatasetType.MATH:
        from .math_dataset import load_math
        return load_math(config.n_eval, config.seed,
                        levels=config.math_levels, subjects=config.math_subjects)
    elif config.type == DatasetType.COUNTDOWN:
        from .countdown import load_countdown
        return load_countdown(config.n_eval, config.seed,
                             num_count=config.countdown_num_count)
    else:
        raise ValueError(f"Unknown dataset type: {config.type}")


__all__ = ["DatasetExample", "create_dataset"]
