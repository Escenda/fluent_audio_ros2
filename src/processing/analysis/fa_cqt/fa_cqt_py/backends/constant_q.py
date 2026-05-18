from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CqtConfig:
    sample_rate: int
    frame_length: int
    hop_length: int
    bin_count: int
    bins_per_octave: int
    f_min_hz: float
    window: str
    normalization: str


@dataclass(frozen=True)
class CqtKernel:
    offset: int
    frequency_hz: float
    window_length: int
    values: np.ndarray


@dataclass(frozen=True)
class CqtResult:
    frame_count: int
    real: np.ndarray
    imag: np.ndarray


class InternalCqtBackend:
    name = "internal_cqt"

    def __init__(self, config: CqtConfig) -> None:
        self.config = _validate_config(config)
        self._kernels = _build_kernels(self.config)
        self.center_frequencies_hz = np.asarray(
            [kernel.frequency_hz for kernel in self._kernels],
            dtype=np.float32,
        )
        self.window_lengths = np.asarray(
            [kernel.window_length for kernel in self._kernels],
            dtype=np.uint32,
        )
        self.f_max_hz = float(self.center_frequencies_hz[-1])

    def compute(self, samples: np.ndarray) -> CqtResult:
        samples = _validate_samples(samples)
        frame_count = _frame_count(
            samples.size,
            self.config.frame_length,
            self.config.hop_length,
        )
        real = np.empty((frame_count, self.config.bin_count), dtype=np.float32)
        imag = np.empty((frame_count, self.config.bin_count), dtype=np.float32)
        for frame_index in range(frame_count):
            start = frame_index * self.config.hop_length
            frame = samples[start : start + self.config.frame_length]
            frame_values = self._compute_frame(frame)
            real[frame_index] = frame_values.real.astype(np.float32)
            imag[frame_index] = frame_values.imag.astype(np.float32)

        if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
            raise RuntimeError("CQT output contains non-finite values")
        return CqtResult(frame_count=frame_count, real=real, imag=imag)

    def _compute_frame(self, frame: np.ndarray) -> np.ndarray:
        values = np.empty(self.config.bin_count, dtype=np.complex128)
        for bin_index, kernel in enumerate(self._kernels):
            segment = frame[kernel.offset : kernel.offset + kernel.window_length]
            values[bin_index] = np.vdot(kernel.values, segment)
        return values


def _validate_config(config: CqtConfig) -> CqtConfig:
    if config.sample_rate <= 0:
        raise RuntimeError("feature.sample_rate must be > 0")
    if config.frame_length <= 1:
        raise RuntimeError("feature.frame_length must be > 1")
    if config.hop_length <= 0:
        raise RuntimeError("feature.hop_length must be > 0")
    if config.hop_length > config.frame_length:
        raise RuntimeError("feature.hop_length must be <= feature.frame_length")
    if config.bin_count <= 0:
        raise RuntimeError("feature.bin_count must be > 0")
    if config.bins_per_octave <= 0:
        raise RuntimeError("feature.bins_per_octave must be > 0")
    if not np.isfinite(config.f_min_hz) or config.f_min_hz <= 0.0:
        raise RuntimeError("feature.f_min_hz must be finite and > 0.0")
    if config.window != "hann":
        raise RuntimeError("feature.window must be hann")
    if config.normalization != "l2":
        raise RuntimeError("feature.normalization must be l2")

    frequencies = _center_frequencies(config)
    nyquist = float(config.sample_rate) / 2.0
    if float(frequencies[-1]) > nyquist:
        raise RuntimeError("feature.bin_count exceeds Nyquist")
    q_factor = _q_factor(config.bins_per_octave)
    longest_kernel = int(np.ceil(q_factor * float(config.sample_rate) / config.f_min_hz))
    if longest_kernel > config.frame_length:
        raise RuntimeError("feature.f_min_hz requires feature.frame_length greater than max CQT kernel")
    return config


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.float32:
        raise ValueError("CQT input samples must be float32")
    if samples.ndim != 1:
        raise ValueError("CQT input samples must be one-dimensional")
    if samples.size == 0:
        raise ValueError("CQT input samples are required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("CQT input samples contain non-finite values")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("CQT input samples must be normalized to [-1.0, 1.0]")
    return samples


def _frame_count(sample_count: int, frame_length: int, hop_length: int) -> int:
    if sample_count < frame_length:
        raise ValueError("CQT input sample count must be >= feature.frame_length")
    if (sample_count - frame_length) % hop_length != 0:
        raise ValueError("CQT input sample count must align to feature.frame_length and feature.hop_length")
    return 1 + ((sample_count - frame_length) // hop_length)


def _build_kernels(config: CqtConfig) -> tuple[CqtKernel, ...]:
    kernels: list[CqtKernel] = []
    q_factor = _q_factor(config.bins_per_octave)
    for frequency in _center_frequencies(config):
        window_length = int(np.ceil(q_factor * float(config.sample_rate) / float(frequency)))
        window = np.hanning(window_length).astype(np.float64)
        time = np.arange(window_length, dtype=np.float64) / float(config.sample_rate)
        sinusoid = np.exp(-2j * np.pi * float(frequency) * time)
        kernel = (window * sinusoid).astype(np.complex128)
        norm = float(np.sqrt(np.sum(np.abs(kernel) ** 2)))
        if norm <= 0.0:
            raise RuntimeError("CQT kernel normalization is zero")
        offset = (config.frame_length - window_length) // 2
        kernels.append(
            CqtKernel(
                offset=offset,
                frequency_hz=float(frequency),
                window_length=window_length,
                values=kernel / norm,
            )
        )
    return tuple(kernels)


def _center_frequencies(config: CqtConfig) -> np.ndarray:
    bin_indices = np.arange(config.bin_count, dtype=np.float64)
    return config.f_min_hz * (2.0 ** (bin_indices / float(config.bins_per_octave)))


def _q_factor(bins_per_octave: int) -> float:
    return 1.0 / ((2.0 ** (1.0 / float(bins_per_octave))) - 1.0)
