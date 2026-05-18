from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameMeterConfig:
    sample_rate: int
    db_floor: float


@dataclass(frozen=True)
class LoudnessResult:
    sample_count: int
    rms: float
    peak: float
    rms_dbfs: float
    peak_dbfs: float
    crest_factor: float


class InternalFrameMeterBackend:
    name = "internal_frame_meter"

    def __init__(self, config: FrameMeterConfig) -> None:
        self.config = _validate_config(config)
        self._floor_amplitude = float(10.0 ** (self.config.db_floor / 20.0))

    def measure(self, samples: np.ndarray) -> LoudnessResult:
        samples = _validate_samples(samples)
        samples64 = samples.astype(np.float64, copy=False)
        squared_mean = float(np.mean(samples64 * samples64))
        rms = math.sqrt(squared_mean)
        peak = float(np.max(np.abs(samples)))
        rms_dbfs = _dbfs(rms, self._floor_amplitude, self.config.db_floor)
        peak_dbfs = _dbfs(peak, self._floor_amplitude, self.config.db_floor)
        crest_factor = 0.0 if rms == 0.0 else peak / rms
        if not all(
            math.isfinite(value)
            for value in (rms, peak, rms_dbfs, peak_dbfs, crest_factor)
        ):
            raise RuntimeError("loudness output contains non-finite values")
        return LoudnessResult(
            sample_count=int(samples.size),
            rms=float(rms),
            peak=peak,
            rms_dbfs=rms_dbfs,
            peak_dbfs=peak_dbfs,
            crest_factor=float(crest_factor),
        )


def _validate_config(config: FrameMeterConfig) -> FrameMeterConfig:
    if config.sample_rate <= 0:
        raise RuntimeError("meter.sample_rate must be > 0")
    if not math.isfinite(config.db_floor) or config.db_floor >= 0.0:
        raise RuntimeError("meter.db_floor must be finite and < 0.0")
    if config.db_floor < -300.0:
        raise RuntimeError("meter.db_floor must be >= -300.0")
    return config


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.float32:
        raise ValueError("loudness input samples must be float32")
    if samples.ndim != 1:
        raise ValueError("loudness input samples must be one-dimensional")
    if samples.size == 0:
        raise ValueError("loudness input samples are required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("loudness input samples contain non-finite values")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("loudness input samples must be normalized to [-1.0, 1.0]")
    return samples


def _dbfs(value: float, floor_amplitude: float, db_floor: float) -> float:
    if value <= floor_amplitude:
        return float(db_floor)
    return float(20.0 * math.log10(value))
