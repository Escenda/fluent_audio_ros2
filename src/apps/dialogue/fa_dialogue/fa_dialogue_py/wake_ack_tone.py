from __future__ import annotations

import math
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class WakeAckToneConfig:
    sample_rate: int = 48000
    channels: int = 1
    duration_ms: int = 260
    fade_ms: int = 34
    gain: float = 0.18
    base_hz: float = 660.0
    lift_hz: float = 260.0
    shimmer_hz: float = 1320.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels != 1:
            raise ValueError("only mono wake ack tones are supported")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if self.fade_ms < 0:
            raise ValueError("fade_ms must be >= 0")
        if self.fade_ms * 2 >= self.duration_ms:
            raise ValueError("fade_ms must be less than half of duration_ms")
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("gain must be in (0.0, 1.0]")
        if self.base_hz <= 0.0 or self.lift_hz < 0.0 or self.shimmer_hz <= 0.0:
            raise ValueError("tone frequencies must be positive")


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _envelope(t: float, *, duration: float, fade: float) -> float:
    if fade <= 0.0:
        return 1.0
    if t < fade:
        return _smoothstep(t / fade)
    remaining = duration - t
    if remaining < fade:
        return _smoothstep(remaining / fade)
    return 1.0


def synthesize_wake_ack_pcm16(config: WakeAckToneConfig) -> bytes:
    """Create a short eased wake acknowledgement earcon as PCM16LE mono."""
    sample_count = int(config.sample_rate * (config.duration_ms / 1000.0))
    duration = config.duration_ms / 1000.0
    fade = config.fade_ms / 1000.0
    phase_main = 0.0
    phase_shimmer = 0.0
    samples = bytearray()
    for index in range(sample_count):
        t = index / config.sample_rate
        progress = t / duration
        lift = _smoothstep(progress)
        main_hz = config.base_hz + (config.lift_hz * lift)
        shimmer_hz = config.shimmer_hz + (config.lift_hz * 0.35 * lift)
        phase_main += (2.0 * math.pi * main_hz) / config.sample_rate
        phase_shimmer += (2.0 * math.pi * shimmer_hz) / config.sample_rate
        env = _envelope(t, duration=duration, fade=fade)
        release_duck = 1.0 - (0.28 * _smoothstep(progress))
        value = (
            math.sin(phase_main) * 0.78
            + math.sin(phase_shimmer) * 0.22
        ) * env * release_duck * config.gain
        clipped = max(-1.0, min(1.0, value))
        samples.extend(struct.pack("<h", int(clipped * 32767.0)))
    return bytes(samples)
