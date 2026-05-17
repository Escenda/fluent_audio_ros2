from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TurnDetectionResult:
    probability: float
    is_end: bool


class TurnDetectorBackend(Protocol):
    name: str
    sample_rate: int
    min_samples: int
    max_samples: int
    model_path: Path

    def detect(self, audio: np.ndarray) -> TurnDetectionResult:
        ...
