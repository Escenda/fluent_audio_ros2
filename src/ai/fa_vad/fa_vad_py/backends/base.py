from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np


@dataclass(frozen=True)
class Float32MonoWindow:
    sample_rate: int
    data: bytes

    def __post_init__(self) -> None:
        if self.sample_rate not in (8000, 16000):
            raise ValueError("Float32MonoWindow sample_rate must be 8000 or 16000")
        if not self.data:
            raise ValueError("Float32MonoWindow data is required")
        if len(self.data) % np.dtype("<f4").itemsize != 0:
            raise ValueError("Float32MonoWindow data must be float32 byte-aligned")
        samples = np.frombuffer(self.data, dtype="<f4")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Float32MonoWindow data contains non-finite samples")
        if np.any(samples < -1.0) or np.any(samples > 1.0):
            raise ValueError("Float32MonoWindow samples must be normalized to [-1.0, 1.0]")


class VADResult(NamedTuple):
    probability: float
    is_speech: bool
    start: bool
    end: bool


VADDecision = VADResult | None


class VADBackend(Protocol):
    name: str

    def update(self, window: Float32MonoWindow) -> VADDecision:
        ...
