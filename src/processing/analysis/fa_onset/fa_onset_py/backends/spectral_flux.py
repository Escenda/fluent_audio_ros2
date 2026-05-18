from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OnsetConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    method: str
    threshold: float
    min_interval_sec: float


@dataclass(frozen=True)
class OnsetResult:
    frame_count: int
    frame_times_sec: np.ndarray
    strengths: np.ndarray
    detected: np.ndarray


class InternalSpectralFluxBackend:
    name = "internal_spectral_flux"

    def __init__(self, config: OnsetConfig) -> None:
        self.config = _validate_config(config)
        self._window = np.hanning(self.config.n_fft).astype(np.float32)

    def compute(self, samples: np.ndarray) -> OnsetResult:
        samples = _validate_samples(samples)
        frame_count = _frame_count(samples.size, self.config.n_fft, self.config.hop_length)
        frames = np.empty((frame_count, self.config.n_fft), dtype=np.float32)
        for frame_index in range(frame_count):
            start = frame_index * self.config.hop_length
            frames[frame_index] = samples[start : start + self.config.n_fft]

        magnitude = np.abs(np.fft.rfft(frames * self._window, axis=1)).astype(np.float32)
        previous = np.zeros_like(magnitude)
        if frame_count > 1:
            previous[1:] = magnitude[:-1]
        strengths = np.sum(np.maximum(magnitude - previous, 0.0), axis=1).astype(np.float32)
        detected = _detect_onsets(strengths, self.config)
        frame_times_sec = _frame_times(frame_count, self.config)

        if not np.all(np.isfinite(strengths)):
            raise RuntimeError("onset strengths contain non-finite values")
        return OnsetResult(
            frame_count=frame_count,
            frame_times_sec=frame_times_sec,
            strengths=strengths,
            detected=detected,
        )


def _validate_config(config: OnsetConfig) -> OnsetConfig:
    if config.sample_rate <= 0:
        raise RuntimeError("feature.sample_rate must be > 0")
    if config.n_fft <= 1:
        raise RuntimeError("feature.n_fft must be > 1")
    if config.hop_length <= 0:
        raise RuntimeError("feature.hop_length must be > 0")
    if config.hop_length > config.n_fft:
        raise RuntimeError("feature.hop_length must be <= feature.n_fft")
    if config.method != "spectral_flux":
        raise RuntimeError("feature.method must be spectral_flux")
    if not np.isfinite(config.threshold) or config.threshold <= 0.0:
        raise RuntimeError("detector.threshold must be finite and > 0.0")
    if not np.isfinite(config.min_interval_sec) or config.min_interval_sec < 0.0:
        raise RuntimeError("detector.min_interval_sec must be finite and >= 0.0")
    return config


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.float32:
        raise ValueError("onset input samples must be float32")
    if samples.ndim != 1:
        raise ValueError("onset input samples must be one-dimensional")
    if samples.size == 0:
        raise ValueError("onset input samples are required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("onset input samples contain non-finite values")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("onset input samples must be normalized to [-1.0, 1.0]")
    return samples


def _frame_count(sample_count: int, n_fft: int, hop_length: int) -> int:
    if sample_count < n_fft:
        raise ValueError("onset input sample count must be >= feature.n_fft")
    if (sample_count - n_fft) % hop_length != 0:
        raise ValueError("onset input sample count must align to feature.n_fft and feature.hop_length")
    return 1 + ((sample_count - n_fft) // hop_length)


def _detect_onsets(strengths: np.ndarray, config: OnsetConfig) -> np.ndarray:
    detected = np.zeros(strengths.shape, dtype=np.bool_)
    min_interval_frames = int(np.ceil(config.min_interval_sec * float(config.sample_rate) / float(config.hop_length)))
    last_detected = -min_interval_frames - 1

    for frame_index, strength in enumerate(strengths):
        if float(strength) < config.threshold:
            continue
        previous = strengths[frame_index - 1] if frame_index > 0 else -np.inf
        next_strength = strengths[frame_index + 1] if frame_index + 1 < strengths.size else -np.inf
        if float(strength) < float(previous) or float(strength) < float(next_strength):
            continue
        if frame_index - last_detected <= min_interval_frames:
            continue
        detected[frame_index] = True
        last_detected = frame_index
    return detected


def _frame_times(frame_count: int, config: OnsetConfig) -> np.ndarray:
    frame_indices = np.arange(frame_count, dtype=np.float32)
    center_offsets = (float(config.n_fft) / 2.0) / float(config.sample_rate)
    hop_sec = float(config.hop_length) / float(config.sample_rate)
    return (frame_indices * hop_sec + center_offsets).astype(np.float32)
