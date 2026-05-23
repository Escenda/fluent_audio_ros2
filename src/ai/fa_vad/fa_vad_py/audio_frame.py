from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioFrameContract:
    source_id: str
    stream_id: str
    sample_rate: int
    channels: int = 1
    encoding: str = "FLOAT32LE"
    bit_depth: int = 32
    layout: str = "interleaved"


def audio_frame_to_float32_mono(msg: object, contract: AudioFrameContract) -> np.ndarray:
    data = getattr(msg, "data")
    if not data:
        raise ValueError("AudioFrame data is required")
    if getattr(msg, "source_id") != contract.source_id:
        raise ValueError("AudioFrame source_id must match expected_source_id")
    if getattr(msg, "stream_id") != contract.stream_id:
        raise ValueError("AudioFrame stream_id must match expected_stream_id")
    if int(getattr(msg, "sample_rate")) != contract.sample_rate:
        raise ValueError(f"AudioFrame sample_rate must be {contract.sample_rate}")
    if int(getattr(msg, "channels")) != contract.channels:
        raise ValueError(f"AudioFrame channels must be {contract.channels}")
    if getattr(msg, "encoding") != contract.encoding:
        raise ValueError(f"AudioFrame encoding must be {contract.encoding}")
    if int(getattr(msg, "bit_depth")) != contract.bit_depth:
        raise ValueError(f"AudioFrame bit_depth must be {contract.bit_depth}")
    if getattr(msg, "layout") != contract.layout:
        raise ValueError(f"AudioFrame layout must be {contract.layout}")
    if len(data) % np.dtype("<f4").itemsize != 0:
        raise ValueError("AudioFrame float32 data length is not byte-aligned")
    samples = np.frombuffer(bytes(data), dtype="<f4")
    if not np.all(np.isfinite(samples)):
        raise ValueError("AudioFrame contains non-finite samples")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("AudioFrame samples must be normalized to [-1.0, 1.0]")
    return samples.astype(np.float32, copy=False)
