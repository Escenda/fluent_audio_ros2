from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fa_vad_py.backends.base import VadProbabilityBackend


@dataclass(frozen=True)
class VadProbabilityFrame:
    probability: float
    window_start_sample: int
    window_end_sample: int


class VadProbabilityStream:
    """Buffers PCM and runs the VAD model on fixed inference windows."""

    def __init__(self, backend: VadProbabilityBackend) -> None:
        self.backend = backend
        self.reset()

    def reset(self) -> None:
        self.backend.reset()
        self.pending = np.empty(0, dtype=np.float32)
        self.processed_samples = 0

    def push(self, samples_float32_16k_mono: np.ndarray) -> list[VadProbabilityFrame]:
        samples = self._validate_samples(samples_float32_16k_mono)
        if samples.size == 0:
            return []
        self.pending = np.concatenate((self.pending, samples))

        frames: list[VadProbabilityFrame] = []
        window = self.backend.window_size_samples
        while self.pending.size >= window:
            frames.append(self._predict_window(self.pending[:window]))
            self.pending = self.pending[window:]
        return frames

    def _predict_window(self, samples: np.ndarray) -> VadProbabilityFrame:
        start_sample = self.processed_samples
        end_sample = start_sample + int(samples.size)
        probability = self.backend.predict_probability(samples)
        self.processed_samples = end_sample
        return VadProbabilityFrame(
            probability=probability,
            window_start_sample=start_sample,
            window_end_sample=end_sample,
        )

    @staticmethod
    def _validate_samples(samples: np.ndarray) -> np.ndarray:
        if not isinstance(samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray")
        if samples.ndim != 1:
            raise ValueError("samples must be mono 1-D float32 PCM")
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples contain non-finite values")
        return np.clip(samples, -1.0, 1.0)
