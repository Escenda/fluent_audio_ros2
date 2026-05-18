from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class AudioEmbeddingRequest:
    samples: np.ndarray
    sample_rate: int
    source_id: str
    stream_id: str

    def __post_init__(self) -> None:
        if int(self.sample_rate) <= 0:
            raise ValueError("AudioEmbeddingRequest sample_rate must be positive")
        if not self.source_id.strip():
            raise ValueError("AudioEmbeddingRequest source_id is required")
        if not self.stream_id.strip():
            raise ValueError("AudioEmbeddingRequest stream_id is required")
        if self.samples.dtype != np.float32:
            raise ValueError("AudioEmbeddingRequest samples must be float32")
        if self.samples.ndim != 1:
            raise ValueError("AudioEmbeddingRequest samples must be one-dimensional")
        if self.samples.size == 0:
            raise ValueError("AudioEmbeddingRequest samples are required")
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("AudioEmbeddingRequest contains non-finite samples")
        if np.any(self.samples < -1.0) or np.any(self.samples > 1.0):
            raise ValueError("AudioEmbeddingRequest samples must be normalized to [-1.0, 1.0]")


@dataclass(frozen=True)
class AudioEmbeddingResult:
    model_id: str
    embedding: np.ndarray

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("AudioEmbeddingResult model_id is required")
        if self.embedding.dtype != np.float32:
            raise ValueError("AudioEmbeddingResult embedding must be float32")
        if self.embedding.ndim != 1:
            raise ValueError("AudioEmbeddingResult embedding must be one-dimensional")
        if self.embedding.size == 0:
            raise ValueError("AudioEmbeddingResult embedding is required")
        if not np.all(np.isfinite(self.embedding)):
            raise ValueError("AudioEmbeddingResult embedding contains non-finite values")


class AudioEmbeddingBackend(Protocol):
    name: str

    def embed(self, request: AudioEmbeddingRequest) -> AudioEmbeddingResult:
        ...
