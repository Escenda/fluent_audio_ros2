from __future__ import annotations

import numpy as np


def validate_node_config(
    *,
    target_sample_rate: int,
    threshold_start: float,
    threshold_end: float,
    hangover_ms: int,
) -> None:
    if target_sample_rate <= 0:
        raise RuntimeError("target_sample_rate must be > 0")
    if threshold_start < 0.0 or threshold_start > 1.0:
        raise RuntimeError("threshold_start must be in [0.0, 1.0]")
    if threshold_end < 0.0 or threshold_end > 1.0:
        raise RuntimeError("threshold_end must be in [0.0, 1.0]")
    if threshold_end > threshold_start:
        raise RuntimeError("threshold_end must be <= threshold_start")
    if hangover_ms <= 0:
        raise RuntimeError("hangover_ms must be > 0")


def validate_qos_depth(depth: int) -> int:
    depth_value = int(depth)
    if depth_value <= 0:
        raise RuntimeError("qos.depth must be > 0")
    return depth_value


def audio_frame_to_float_samples(
    *,
    data: bytes,
    source_id: str,
    stream_id: str,
    expected_source_id: str,
    expected_stream_id: str,
    encoding: str,
    layout: str,
    channels: int,
    bit_depth: int,
) -> np.ndarray:
    if not data:
        raise ValueError("AudioFrame data is required")
    if not source_id or not stream_id:
        raise ValueError("AudioFrame source_id and stream_id are required")
    if not expected_source_id:
        raise ValueError("expected_source_id is required")
    if not expected_stream_id:
        raise ValueError("expected_stream_id is required")
    if source_id != expected_source_id:
        raise ValueError("AudioFrame source_id must match expected_source_id")
    if stream_id != expected_stream_id:
        raise ValueError("AudioFrame stream_id must match input_topic")
    if layout != "interleaved":
        raise ValueError(f"AudioFrame layout must be interleaved, got {layout}")
    if int(channels) != 1:
        raise ValueError(f"AudioFrame channels must be 1, got {channels}")
    if encoding != "FLOAT32LE":
        raise ValueError(f"AudioFrame encoding must be FLOAT32LE, got {encoding}")
    if int(bit_depth) != 32:
        raise ValueError(f"AudioFrame bit_depth must be 32, got {bit_depth}")
    if len(data) % np.dtype("<f4").itemsize != 0:
        raise ValueError("AudioFrame float32 data length is not byte-aligned")

    samples = np.frombuffer(data, dtype="<f4")
    if not np.all(np.isfinite(samples)):
        raise ValueError("AudioFrame contains non-finite samples")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("AudioFrame samples must be normalized to [-1.0, 1.0]")

    return samples
