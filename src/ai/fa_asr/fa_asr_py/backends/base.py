from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class AsrRequest:
    session_id: str
    user_turn_id: int
    samples: np.ndarray
    sample_rate: int


class AsrBackend(Protocol):
    name: str

    def transcribe(self, request: AsrRequest) -> str:
        ...
