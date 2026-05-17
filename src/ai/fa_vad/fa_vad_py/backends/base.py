from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol


@dataclass(frozen=True)
class Pcm16MonoWindow:
    sample_rate: int
    data: bytes

    def __post_init__(self) -> None:
        if self.sample_rate not in (8000, 16000):
            raise ValueError("Pcm16MonoWindow sample_rate must be 8000 or 16000")
        if not self.data:
            raise ValueError("Pcm16MonoWindow data is required")
        if len(self.data) % 2 != 0:
            raise ValueError("Pcm16MonoWindow data must be PCM16 byte-aligned")


class VADResult(NamedTuple):
    probability: float
    is_speech: bool
    start: bool
    end: bool


VADDecision = VADResult | None


class VADBackend(Protocol):
    name: str

    def update(self, window: Pcm16MonoWindow) -> VADDecision:
        ...
