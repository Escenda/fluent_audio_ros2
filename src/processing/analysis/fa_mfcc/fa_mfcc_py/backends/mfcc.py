from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MfccConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int
    n_mfcc: int
    f_min_hz: float
    f_max_hz: float
    log_floor: float
    dct_type: str
    normalization: str


@dataclass(frozen=True)
class MfccResult:
    frame_count: int
    values: np.ndarray


class InternalMfccBackend:
    name = "internal_mfcc"

    def __init__(self, config: MfccConfig) -> None:
        self.config = _validate_config(config)
        self._window = np.hanning(self.config.n_fft).astype(np.float32)
        self._mel_filters = _build_mel_filterbank(self.config)
        self._dct_basis = _build_dct2_basis(self.config)

    def compute(self, samples: np.ndarray) -> MfccResult:
        samples = _validate_samples(samples)
        frame_count = _frame_count(samples.size, self.config.n_fft, self.config.hop_length)
        frames = np.empty((frame_count, self.config.n_fft), dtype=np.float32)
        for frame_index in range(frame_count):
            start = frame_index * self.config.hop_length
            frames[frame_index] = samples[start : start + self.config.n_fft]

        spectrum = np.fft.rfft(frames * self._window, axis=1)
        power = np.abs(spectrum).astype(np.float32) ** 2
        mel_power = power @ self._mel_filters.T
        log_mel = np.log(np.maximum(mel_power, self.config.log_floor)).astype(np.float32)
        mfcc = (log_mel @ self._dct_basis.T).astype(np.float32)
        if not np.all(np.isfinite(mfcc)):
            raise RuntimeError("MFCC output contains non-finite values")
        return MfccResult(frame_count=frame_count, values=mfcc)


def _validate_config(config: MfccConfig) -> MfccConfig:
    if config.sample_rate <= 0:
        raise RuntimeError("feature.sample_rate must be > 0")
    if config.n_fft <= 1:
        raise RuntimeError("feature.n_fft must be > 1")
    if config.hop_length <= 0:
        raise RuntimeError("feature.hop_length must be > 0")
    if config.hop_length > config.n_fft:
        raise RuntimeError("feature.hop_length must be <= feature.n_fft")
    if config.n_mels <= 0:
        raise RuntimeError("feature.n_mels must be > 0")
    if config.n_mfcc <= 0:
        raise RuntimeError("feature.n_mfcc must be > 0")
    if config.n_mfcc > config.n_mels:
        raise RuntimeError("feature.n_mfcc must be <= feature.n_mels")
    if not np.isfinite(config.f_min_hz) or config.f_min_hz < 0.0:
        raise RuntimeError("feature.f_min_hz must be finite and >= 0.0")
    nyquist = float(config.sample_rate) / 2.0
    if not np.isfinite(config.f_max_hz) or config.f_max_hz <= config.f_min_hz:
        raise RuntimeError("feature.f_max_hz must be finite and greater than feature.f_min_hz")
    if config.f_max_hz > nyquist:
        raise RuntimeError("feature.f_max_hz must be <= Nyquist")
    if not np.isfinite(config.log_floor) or config.log_floor <= 0.0:
        raise RuntimeError("feature.log_floor must be finite and > 0.0")
    if config.dct_type != "dct2":
        raise RuntimeError("feature.dct_type must be dct2")
    if config.normalization != "ortho":
        raise RuntimeError("feature.normalization must be ortho")
    return config


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.float32:
        raise ValueError("MFCC input samples must be float32")
    if samples.ndim != 1:
        raise ValueError("MFCC input samples must be one-dimensional")
    if samples.size == 0:
        raise ValueError("MFCC input samples are required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("MFCC input samples contain non-finite values")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("MFCC input samples must be normalized to [-1.0, 1.0]")
    return samples


def _frame_count(sample_count: int, n_fft: int, hop_length: int) -> int:
    if sample_count < n_fft:
        raise ValueError("MFCC input sample count must be >= feature.n_fft")
    if (sample_count - n_fft) % hop_length != 0:
        raise ValueError("MFCC input sample count must align to feature.n_fft and feature.hop_length")
    return 1 + ((sample_count - n_fft) // hop_length)


def _build_mel_filterbank(config: MfccConfig) -> np.ndarray:
    n_fft_bins = (config.n_fft // 2) + 1
    mel_min = _hz_to_mel(config.f_min_hz)
    mel_max = _hz_to_mel(config.f_max_hz)
    mel_points = np.linspace(mel_min, mel_max, config.n_mels + 2, dtype=np.float64)
    hz_points = _mel_to_hz(mel_points)
    fft_freqs = np.linspace(0.0, float(config.sample_rate) / 2.0, n_fft_bins)

    filters = np.zeros((config.n_mels, n_fft_bins), dtype=np.float32)
    for mel_index in range(config.n_mels):
        left_hz = hz_points[mel_index]
        center_hz = hz_points[mel_index + 1]
        right_hz = hz_points[mel_index + 2]
        left_slope = (fft_freqs - left_hz) / max(center_hz - left_hz, 1e-12)
        right_slope = (right_hz - fft_freqs) / max(right_hz - center_hz, 1e-12)
        filters[mel_index] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    enorm = 2.0 / np.maximum(hz_points[2 : config.n_mels + 2] - hz_points[: config.n_mels], 1e-12)
    filters *= enorm[:, np.newaxis].astype(np.float32)
    return filters


def _build_dct2_basis(config: MfccConfig) -> np.ndarray:
    mel_indices = np.arange(config.n_mels, dtype=np.float64)
    coeff_indices = np.arange(config.n_mfcc, dtype=np.float64)[:, np.newaxis]
    basis = np.cos((np.pi / float(config.n_mels)) * (mel_indices + 0.5) * coeff_indices)
    basis[0, :] *= np.sqrt(1.0 / float(config.n_mels))
    if config.n_mfcc > 1:
        basis[1:, :] *= np.sqrt(2.0 / float(config.n_mels))
    return basis.astype(np.float32)


def _hz_to_mel(hz: float) -> float:
    return float(2595.0 * np.log10(1.0 + hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * ((10.0 ** (mel / 2595.0)) - 1.0)
