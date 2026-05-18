from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StftConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    window: str


@dataclass(frozen=True)
class StftResult:
    frame_count: int
    bin_count: int
    real: np.ndarray
    imag: np.ndarray


class InternalStftBackend:
    name = "internal_stft"

    def __init__(self, config: StftConfig) -> None:
        self.config = _validate_config(config)
        self._window = _build_window(self.config.window, self.config.n_fft)

    def compute(self, samples: np.ndarray) -> StftResult:
        samples = _validate_samples(samples)
        frame_count = _frame_count(samples.size, self.config.n_fft, self.config.hop_length)
        frames = np.empty((frame_count, self.config.n_fft), dtype=np.float32)
        for frame_index in range(frame_count):
            start = frame_index * self.config.hop_length
            frames[frame_index] = samples[start : start + self.config.n_fft]

        spectrum = np.fft.rfft(frames * self._window, axis=1)
        real = spectrum.real.astype(np.float32)
        imag = spectrum.imag.astype(np.float32)
        if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
            raise RuntimeError("STFT output contains non-finite values")
        return StftResult(
            frame_count=frame_count,
            bin_count=real.shape[1],
            real=real,
            imag=imag,
        )


def _validate_config(config: StftConfig) -> StftConfig:
    if config.sample_rate <= 0:
        raise RuntimeError("feature.sample_rate must be > 0")
    if config.n_fft <= 1:
        raise RuntimeError("feature.n_fft must be > 1")
    if config.hop_length <= 0:
        raise RuntimeError("feature.hop_length must be > 0")
    if config.hop_length > config.n_fft:
        raise RuntimeError("feature.hop_length must be <= feature.n_fft")
    if config.window not in ("hann", "rectangular"):
        raise RuntimeError("feature.window must be hann or rectangular")
    return config


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.float32:
        raise ValueError("STFT input samples must be float32")
    if samples.ndim != 1:
        raise ValueError("STFT input samples must be one-dimensional")
    if samples.size == 0:
        raise ValueError("STFT input samples are required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("STFT input samples contain non-finite values")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("STFT input samples must be normalized to [-1.0, 1.0]")
    return samples


def _frame_count(sample_count: int, n_fft: int, hop_length: int) -> int:
    if sample_count < n_fft:
        raise ValueError("STFT input sample count must be >= feature.n_fft")
    if (sample_count - n_fft) % hop_length != 0:
        raise ValueError("STFT input sample count must align to feature.n_fft and feature.hop_length")
    return 1 + ((sample_count - n_fft) // hop_length)


def _build_window(window: str, n_fft: int) -> np.ndarray:
    if window == "hann":
        return np.hanning(n_fft).astype(np.float32)
    if window == "rectangular":
        return np.ones(n_fft, dtype=np.float32)
    raise RuntimeError("feature.window must be hann or rectangular")
