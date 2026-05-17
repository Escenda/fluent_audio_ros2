from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SynthesizedAudio:
    audio_bytes: bytes
    encoding: str
    sample_rate: int
    channels: int
    bit_depth: int


class TextToSpeechBackend(Protocol):
    name: str

    def synthesize(self, text: str, voice_id: str) -> SynthesizedAudio:
        ...
