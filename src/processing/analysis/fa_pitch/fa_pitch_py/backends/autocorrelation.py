from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    method: str
    f_min_hz: float
    f_max_hz: float
    confidence_threshold: float


@dataclass(frozen=True)
class PitchResult:
    frame_count: int
    frame_times_sec: np.ndarray
    frequencies_hz: np.ndarray
    confidence: np.ndarray
    voiced: np.ndarray


class InternalAutocorrelationBackend:
    name = "internal_autocorrelation"

    def __init__(self, config: PitchConfig) -> None:
        self.config = _validate_config(config)
        self._window = np.hanning(self.config.n_fft).astype(np.float32)
        self._min_lag = int(np.floor(float(self.config.sample_rate) / self.config.f_max_hz))
        self._max_lag = int(np.ceil(float(self.config.sample_rate) / self.config.f_min_hz))

    def compute(self, samples: np.ndarray) -> PitchResult:
        samples = _validate_samples(samples)
        frame_count = _frame_count(samples.size, self.config.n_fft, self.config.hop_length)
        frequencies_hz = np.zeros(frame_count, dtype=np.float32)
        confidence = np.zeros(frame_count, dtype=np.float32)
        voiced = np.zeros(frame_count, dtype=np.bool_)

        for frame_index in range(frame_count):
            start = frame_index * self.config.hop_length
            frame = samples[start : start + self.config.n_fft]
            frequency_hz, frame_confidence = self._estimate_frame(frame)
            frequencies_hz[frame_index] = frequency_hz
            confidence[frame_index] = frame_confidence
            voiced[frame_index] = frame_confidence >= self.config.confidence_threshold

        frame_times_sec = _frame_times(frame_count, self.config)
        if not np.all(np.isfinite(frequencies_hz)) or not np.all(np.isfinite(confidence)):
            raise RuntimeError("pitch output contains non-finite values")
        return PitchResult(
            frame_count=frame_count,
            frame_times_sec=frame_times_sec,
            frequencies_hz=frequencies_hz,
            confidence=confidence,
            voiced=voiced,
        )

    def _estimate_frame(self, frame: np.ndarray) -> tuple[float, float]:
        centered = frame - np.mean(frame, dtype=np.float32)
        windowed = centered * self._window
        autocorr = np.correlate(windowed, windowed, mode="full")[self.config.n_fft - 1 :]
        energy = float(autocorr[0])
        if energy <= 1.0e-12:
            return 0.0, 0.0

        search = autocorr[self._min_lag : self._max_lag + 1]
        best_offset = int(np.argmax(search))
        best_lag = self._min_lag + best_offset
        peak_ratio = max(float(search[best_offset]) / energy, 0.0)
        frequency_hz = float(self.config.sample_rate) / float(best_lag)
        return frequency_hz, min(peak_ratio, 1.0)


def _validate_config(config: PitchConfig) -> PitchConfig:
    if config.sample_rate <= 0:
        raise RuntimeError("feature.sample_rate must be > 0")
    if config.n_fft <= 1:
        raise RuntimeError("feature.n_fft must be > 1")
    if config.hop_length <= 0:
        raise RuntimeError("feature.hop_length must be > 0")
    if config.hop_length > config.n_fft:
        raise RuntimeError("feature.hop_length must be <= feature.n_fft")
    if config.method != "autocorrelation":
        raise RuntimeError("feature.method must be autocorrelation")
    if not np.isfinite(config.f_min_hz) or config.f_min_hz <= 0.0:
        raise RuntimeError("feature.f_min_hz must be finite and > 0.0")
    if not np.isfinite(config.f_max_hz) or config.f_max_hz <= config.f_min_hz:
        raise RuntimeError("feature.f_max_hz must be finite and greater than feature.f_min_hz")
    nyquist = float(config.sample_rate) / 2.0
    if config.f_max_hz > nyquist:
        raise RuntimeError("feature.f_max_hz must be <= Nyquist")
    max_lag = int(np.ceil(float(config.sample_rate) / config.f_min_hz))
    min_lag = int(np.floor(float(config.sample_rate) / config.f_max_hz))
    if min_lag < 1:
        raise RuntimeError("feature.f_max_hz is too high for sample_rate")
    if max_lag >= config.n_fft:
        raise RuntimeError("feature.f_min_hz requires feature.n_fft greater than max pitch lag")
    if not np.isfinite(config.confidence_threshold):
        raise RuntimeError("detector.confidence_threshold must be finite")
    if config.confidence_threshold < 0.0 or config.confidence_threshold > 1.0:
        raise RuntimeError("detector.confidence_threshold must be in [0.0, 1.0]")
    return config


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.float32:
        raise ValueError("pitch input samples must be float32")
    if samples.ndim != 1:
        raise ValueError("pitch input samples must be one-dimensional")
    if samples.size == 0:
        raise ValueError("pitch input samples are required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("pitch input samples contain non-finite values")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("pitch input samples must be normalized to [-1.0, 1.0]")
    return samples


def _frame_count(sample_count: int, n_fft: int, hop_length: int) -> int:
    if sample_count < n_fft:
        raise ValueError("pitch input sample count must be >= feature.n_fft")
    if (sample_count - n_fft) % hop_length != 0:
        raise ValueError("pitch input sample count must align to feature.n_fft and feature.hop_length")
    return 1 + ((sample_count - n_fft) // hop_length)


def _frame_times(frame_count: int, config: PitchConfig) -> np.ndarray:
    frame_indices = np.arange(frame_count, dtype=np.float32)
    center_offsets = (float(config.n_fft) / 2.0) / float(config.sample_rate)
    hop_sec = float(config.hop_length) / float(config.sample_rate)
    return (frame_indices * hop_sec + center_offsets).astype(np.float32)
