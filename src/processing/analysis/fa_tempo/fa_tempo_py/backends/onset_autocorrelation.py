from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TempoConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    method: str
    bpm_min: float
    bpm_max: float
    confidence_threshold: float


@dataclass(frozen=True)
class TempoResult:
    frame_count: int
    frame_times_sec: np.ndarray
    onset_envelope: np.ndarray
    beats: np.ndarray
    tempo_bpm: float
    confidence: float
    beat_period_frames: int
    tempo_detected: bool


class InternalOnsetAutocorrelationBackend:
    name = "internal_onset_autocorrelation"

    def __init__(self, config: TempoConfig) -> None:
        self.config = _validate_config(config)
        self._window = np.hanning(self.config.n_fft).astype(np.float32)

    def compute(self, samples: np.ndarray) -> TempoResult:
        samples = _validate_samples(samples)
        frame_count = _frame_count(samples.size, self.config.n_fft, self.config.hop_length)
        envelope = self._onset_envelope(samples, frame_count)
        tempo_bpm, confidence, beat_period_frames = _estimate_tempo(envelope, self.config)
        tempo_detected = confidence >= self.config.confidence_threshold
        beats = _detect_beats(envelope, beat_period_frames, tempo_detected)
        if not tempo_detected:
            tempo_bpm = 0.0
            beat_period_frames = 0

        frame_times_sec = _frame_times(frame_count, self.config)
        if not np.all(np.isfinite(envelope)) or not np.isfinite(tempo_bpm) or not np.isfinite(confidence):
            raise RuntimeError("tempo output contains non-finite values")
        return TempoResult(
            frame_count=frame_count,
            frame_times_sec=frame_times_sec,
            onset_envelope=envelope,
            beats=beats,
            tempo_bpm=float(tempo_bpm),
            confidence=float(confidence),
            beat_period_frames=int(beat_period_frames),
            tempo_detected=bool(tempo_detected),
        )

    def _onset_envelope(self, samples: np.ndarray, frame_count: int) -> np.ndarray:
        frames = np.empty((frame_count, self.config.n_fft), dtype=np.float32)
        for frame_index in range(frame_count):
            start = frame_index * self.config.hop_length
            frames[frame_index] = samples[start : start + self.config.n_fft]

        magnitude = np.abs(np.fft.rfft(frames * self._window, axis=1)).astype(np.float32)
        previous = np.zeros_like(magnitude)
        if frame_count > 1:
            previous[1:] = magnitude[:-1]
        envelope = np.sum(np.maximum(magnitude - previous, 0.0), axis=1).astype(np.float32)
        return envelope


def _validate_config(config: TempoConfig) -> TempoConfig:
    if config.sample_rate <= 0:
        raise RuntimeError("feature.sample_rate must be > 0")
    if config.n_fft <= 1:
        raise RuntimeError("feature.n_fft must be > 1")
    if config.hop_length <= 0:
        raise RuntimeError("feature.hop_length must be > 0")
    if config.hop_length > config.n_fft:
        raise RuntimeError("feature.hop_length must be <= feature.n_fft")
    if config.method != "onset_autocorrelation":
        raise RuntimeError("feature.method must be onset_autocorrelation")
    if not np.isfinite(config.bpm_min) or config.bpm_min <= 0.0:
        raise RuntimeError("tempo.bpm_min must be finite and > 0.0")
    if not np.isfinite(config.bpm_max) or config.bpm_max <= config.bpm_min:
        raise RuntimeError("tempo.bpm_max must be finite and greater than tempo.bpm_min")
    min_period_frames = 60.0 * float(config.sample_rate) / (config.bpm_max * float(config.hop_length))
    max_period_frames = 60.0 * float(config.sample_rate) / (config.bpm_min * float(config.hop_length))
    if min_period_frames < 1.0:
        raise RuntimeError("tempo.bpm_max is too high for sample_rate and hop_length")
    if max_period_frames <= min_period_frames:
        raise RuntimeError("tempo BPM range does not produce a searchable lag range")
    if not np.isfinite(config.confidence_threshold):
        raise RuntimeError("tempo.confidence_threshold must be finite")
    if config.confidence_threshold < 0.0 or config.confidence_threshold > 1.0:
        raise RuntimeError("tempo.confidence_threshold must be in [0.0, 1.0]")
    return config


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.float32:
        raise ValueError("tempo input samples must be float32")
    if samples.ndim != 1:
        raise ValueError("tempo input samples must be one-dimensional")
    if samples.size == 0:
        raise ValueError("tempo input samples are required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("tempo input samples contain non-finite values")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("tempo input samples must be normalized to [-1.0, 1.0]")
    return samples


def _frame_count(sample_count: int, n_fft: int, hop_length: int) -> int:
    if sample_count < n_fft:
        raise ValueError("tempo input sample count must be >= feature.n_fft")
    if (sample_count - n_fft) % hop_length != 0:
        raise ValueError("tempo input sample count must align to feature.n_fft and feature.hop_length")
    return 1 + ((sample_count - n_fft) // hop_length)


def _estimate_tempo(envelope: np.ndarray, config: TempoConfig) -> tuple[float, float, int]:
    centered = envelope.astype(np.float64) - float(np.mean(envelope, dtype=np.float64))
    energy = float(np.dot(centered, centered))
    if energy <= 1.0e-12:
        return 0.0, 0.0, 0

    min_lag = int(np.ceil(60.0 * float(config.sample_rate) / (config.bpm_max * float(config.hop_length))))
    max_lag = int(np.floor(60.0 * float(config.sample_rate) / (config.bpm_min * float(config.hop_length))))
    max_lag = min(max_lag, envelope.size - 1)
    if max_lag < min_lag:
        return 0.0, 0.0, 0

    scores = []
    for lag in range(min_lag, max_lag + 1):
        score = float(np.dot(centered[:-lag], centered[lag:]) / energy)
        scores.append(max(score, 0.0))
    score_array = np.asarray(scores, dtype=np.float64)
    best_offset = int(np.argmax(score_array))
    best_score = float(score_array[best_offset])
    best_lag = min_lag + best_offset
    tempo_bpm = 60.0 * float(config.sample_rate) / (float(best_lag) * float(config.hop_length))
    return tempo_bpm, min(best_score, 1.0), best_lag


def _detect_beats(envelope: np.ndarray, beat_period_frames: int, tempo_detected: bool) -> np.ndarray:
    beats = np.zeros(envelope.shape, dtype=np.bool_)
    if not tempo_detected or beat_period_frames <= 0 or envelope.size == 0:
        return beats

    anchor_search_width = min(beat_period_frames, envelope.size)
    anchor = int(np.argmax(envelope[:anchor_search_width]))
    for frame_index in range(anchor, envelope.size, beat_period_frames):
        beats[frame_index] = True
    return beats


def _frame_times(frame_count: int, config: TempoConfig) -> np.ndarray:
    frame_indices = np.arange(frame_count, dtype=np.float32)
    center_offsets = (float(config.n_fft) / 2.0) / float(config.sample_rate)
    hop_sec = float(config.hop_length) / float(config.sample_rate)
    return (frame_indices * hop_sec + center_offsets).astype(np.float32)
