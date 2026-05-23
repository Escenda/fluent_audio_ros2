from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class VadBackendSettings:
    name: str
    model_path: str
    sample_rate: int
    execution_provider: str
    inter_op_num_threads: int
    intra_op_num_threads: int


class VadProbabilityBackend(Protocol):
    name: str
    sample_rate: int
    window_size_samples: int

    def reset(self) -> None:
        ...

    def predict_probability(self, samples: np.ndarray) -> float:
        ...
