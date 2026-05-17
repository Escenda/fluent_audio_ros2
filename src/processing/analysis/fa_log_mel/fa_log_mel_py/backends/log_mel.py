from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogMelConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int
    f_min_hz: float
    f_max_hz: float
    log_floor: float


@dataclass(frozen=True)
class LogMelResult:
    frame_count: int
    values: np.ndarray


class InternalLogMelBackend:
    name = "internal_log_mel"

    def __init__(self, config: LogMelConfig) -> None:
        self.config = _validate_config(config)
        self._window = np.hanning(self.config.n_fft).astype(np.float32)
        self._mel_filters = _build_mel_filterbank(self.config)

    def compute(self, samples: np.ndarray) -> LogMelResult:
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
        if not np.all(np.isfinite(log_mel)):
            raise RuntimeError("log-mel output contains non-finite values")
        return LogMelResult(frame_count=frame_count, values=log_mel)


def _validate_config(config: LogMelConfig) -> LogMelConfig:
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
    if not np.isfinite(config.f_min_hz) or config.f_min_hz < 0.0:
        raise RuntimeError("feature.f_min_hz must be finite and >= 0.0")
    nyquist = float(config.sample_rate) / 2.0
    if not np.isfinite(config.f_max_hz) or config.f_max_hz <= config.f_min_hz:
        raise RuntimeError("feature.f_max_hz must be finite and greater than feature.f_min_hz")
    if config.f_max_hz > nyquist:
        raise RuntimeError("feature.f_max_hz must be <= Nyquist")
    if not np.isfinite(config.log_floor) or config.log_floor <= 0.0:
        raise RuntimeError("feature.log_floor must be finite and > 0.0")
    return config


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.float32:
        raise ValueError("log-mel input samples must be float32")
    if samples.ndim != 1:
        raise ValueError("log-mel input samples must be one-dimensional")
    if samples.size == 0:
        raise ValueError("log-mel input samples are required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("log-mel input samples contain non-finite values")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("log-mel input samples must be normalized to [-1.0, 1.0]")
    return samples


def _frame_count(sample_count: int, n_fft: int, hop_length: int) -> int:
    if sample_count < n_fft:
        raise ValueError("log-mel input sample count must be >= feature.n_fft")
    return 1 + ((sample_count - n_fft) // hop_length)


def _build_mel_filterbank(config: LogMelConfig) -> np.ndarray:
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


def _hz_to_mel(hz: float) -> float:
    return float(2595.0 * np.log10(1.0 + hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * ((10.0 ** (mel / 2595.0)) - 1.0)
