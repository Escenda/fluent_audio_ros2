from __future__ import annotations

from typing import NamedTuple, Protocol


class VADResult(NamedTuple):
    probability: float
    is_speech: bool
    start: bool
    end: bool


class VADBackend(Protocol):
    name: str

    def update(self, pcm16_bytes: bytes) -> VADResult:
        ...
