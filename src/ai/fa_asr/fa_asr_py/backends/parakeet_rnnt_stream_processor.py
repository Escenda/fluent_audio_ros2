from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ContextSize:
    left: int
    chunk: int
    right: int

    def total(self) -> int:
        return self.left + self.chunk + self.right


@dataclass(frozen=True)
class ParakeetStreamConfig:
    model_name: str | None = None
    model_path: str | None = None
    device: str = "cuda"
    compute_dtype: str = "bfloat16"
    sample_rate: int = 16000
    left_context_secs: float = 10.0
    chunk_secs: float = 2.0
    right_context_secs: float = 2.0
    att_context_size_as_chunk: bool = True

    def __post_init__(self) -> None:
        if not self.model_name and not self.model_path:
            raise ValueError("model_name or model_path is required")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.left_context_secs < 0.0:
            raise ValueError("left_context_secs must be >= 0")
        if self.chunk_secs <= 0.0:
            raise ValueError("chunk_secs must be > 0")
        if self.right_context_secs < 0.0:
            raise ValueError("right_context_secs must be >= 0")


@dataclass(frozen=True)
class TranscriptSnapshot:
    text: str
    accepted_samples: int
    complete: bool


@dataclass(frozen=True)
class DecodeStepResult:
    chunk_hypotheses: Any
    decoder_state: Any


@dataclass(frozen=True)
class EncodedAudio:
    output: Any
    output_lengths: Any


class StreamingAudioBuffer(Protocol):
    context_size: Any
    context_size_batch: Any
    samples: Any

    def add_audio_batch(
        self,
        samples: np.ndarray,
        *,
        audio_lengths: Any,
        is_last_chunk: bool,
        is_last_chunk_batch: Any,
    ) -> None:
        ...


class ParakeetRnntRuntime(Protocol):
    sample_rate: int
    feature_stride_sec: float
    encoder_subsampling_factor: int
    uses_chunked_limited_attention_with_right_context: bool

    def configure_for_streaming(self) -> None:
        ...

    def set_default_attention_context(self, context_encoder_frames: ContextSize) -> None:
        ...

    def create_buffer(self, context_samples: ContextSize) -> StreamingAudioBuffer:
        ...

    def make_audio_lengths(self, sample_count: int) -> Any:
        ...

    def make_last_chunk_batch(self, is_last_chunk: bool) -> Any:
        ...

    def encode(self, buffer: StreamingAudioBuffer) -> EncodedAudio:
        ...

    def subsample_context(self, context: Any, *, factor: int) -> Any:
        ...

    def trim_left_context(self, encoded_output: Any, *, left_frames: int) -> Any:
        ...

    def output_length(self, encoded: EncodedAudio, context_batch: Any, *, is_last_chunk: bool) -> Any:
        ...

    def decode(
        self,
        encoded_output: Any,
        output_lengths: Any,
        *,
        previous_state: Any,
    ) -> DecodeStepResult:
        ...

    def merge_hypotheses(self, current_hypotheses: Any, chunk_hypotheses: Any) -> Any:
        ...

    def hypotheses_to_text(self, hypotheses: Any) -> str:
        ...


class ParakeetRnntStreamProcessor:
    """Pure Parakeet RNNT streaming model processor.

    This object deliberately does not know about ROS, yaml pipeline definitions,
    VAD/KWS/TD, turn/session IDs, event emission, publishing, or callbacks.
    """

    def __init__(
        self,
        config: ParakeetStreamConfig,
        *,
        runtime: ParakeetRnntRuntime | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime if runtime is not None else self._load_runtime(config)
        self.runtime.configure_for_streaming()
        self._validate_runtime()
        self.context_encoder_frames = self._context_encoder_frames()
        self.context_samples = self._context_samples()
        self.encoder_frame2audio_samples = self._encoder_frame2audio_samples()
        self._configure_attention_context()
        self.reset()

    def reset(self) -> None:
        self.pending = np.empty(0, dtype=np.float32)
        self.buffer = self.runtime.create_buffer(self.context_samples)
        self.decoder_state: Any = None
        self.current_hypotheses: Any = None
        self.accepted_samples = 0
        self.decode_steps = 0

    def push(self, samples_float32_16k_mono: np.ndarray) -> list[TranscriptSnapshot]:
        samples = self._validate_samples(samples_float32_16k_mono)
        if samples.size == 0:
            return []
        self.pending = np.concatenate((self.pending, samples))

        snapshots: list[TranscriptSnapshot] = []
        while self.pending.size >= self._next_required_samples():
            required = self._next_required_samples()
            snapshots.append(self._decode_step(self.pending[:required], complete=False))
            self.pending = self.pending[required:]
        return snapshots

    def finish(self) -> TranscriptSnapshot:
        if self.pending.size > 0:
            snapshot = self._decode_step(self.pending, complete=True)
            self.pending = np.empty(0, dtype=np.float32)
            return snapshot
        return TranscriptSnapshot(
            text=self._current_text(),
            accepted_samples=self.accepted_samples,
            complete=True,
        )

    def _next_required_samples(self) -> int:
        if self.decode_steps == 0:
            return self.context_samples.chunk + self.context_samples.right
        return self.context_samples.chunk

    def _decode_step(self, samples: np.ndarray, *, complete: bool) -> TranscriptSnapshot:
        audio_lengths = self.runtime.make_audio_lengths(samples.size)
        last_chunk_batch = self.runtime.make_last_chunk_batch(complete)
        self.buffer.add_audio_batch(
            samples,
            audio_lengths=audio_lengths,
            is_last_chunk=complete,
            is_last_chunk_batch=last_chunk_batch,
        )

        encoded = self.runtime.encode(self.buffer)
        encoder_context = self.runtime.subsample_context(
            self.buffer.context_size,
            factor=self.encoder_frame2audio_samples,
        )
        encoder_context_batch = self.runtime.subsample_context(
            self.buffer.context_size_batch,
            factor=self.encoder_frame2audio_samples,
        )
        encoded_output = self.runtime.trim_left_context(
            encoded.output,
            left_frames=int(encoder_context.left),
        )
        output_lengths = self.runtime.output_length(
            EncodedAudio(output=encoded_output, output_lengths=encoded.output_lengths),
            encoder_context_batch,
            is_last_chunk=complete,
        )
        decoded = self.runtime.decode(
            encoded_output,
            output_lengths,
            previous_state=self.decoder_state,
        )
        self.decoder_state = decoded.decoder_state
        self.current_hypotheses = self.runtime.merge_hypotheses(
            self.current_hypotheses,
            decoded.chunk_hypotheses,
        )
        self.decode_steps += 1
        self.accepted_samples += int(samples.size)
        return TranscriptSnapshot(
            text=self._current_text(),
            accepted_samples=self.accepted_samples,
            complete=complete,
        )

    def _current_text(self) -> str:
        if self.current_hypotheses is None:
            return ""
        return self.runtime.hypotheses_to_text(self.current_hypotheses)

    def _context_encoder_frames(self) -> ContextSize:
        features_per_sec = 1.0 / self.runtime.feature_stride_sec
        factor = self.runtime.encoder_subsampling_factor
        return ContextSize(
            left=int(self.config.left_context_secs * features_per_sec / factor),
            chunk=int(self.config.chunk_secs * features_per_sec / factor),
            right=int(self.config.right_context_secs * features_per_sec / factor),
        )

    def _context_samples(self) -> ContextSize:
        factor = self.runtime.encoder_subsampling_factor
        features_frame2audio_samples = self._features_frame2audio_samples()
        return ContextSize(
            left=self.context_encoder_frames.left * factor * features_frame2audio_samples,
            chunk=self.context_encoder_frames.chunk * factor * features_frame2audio_samples,
            right=self.context_encoder_frames.right * factor * features_frame2audio_samples,
        )

    def _features_frame2audio_samples(self) -> int:
        samples = int(self.runtime.sample_rate * self.runtime.feature_stride_sec)
        factor = self.runtime.encoder_subsampling_factor
        return (samples // factor) * factor

    def _encoder_frame2audio_samples(self) -> int:
        return self._features_frame2audio_samples() * self.runtime.encoder_subsampling_factor

    def _configure_attention_context(self) -> None:
        if not self.config.att_context_size_as_chunk:
            return
        if not self.runtime.uses_chunked_limited_attention_with_right_context:
            return
        self.runtime.set_default_attention_context(self.context_encoder_frames)

    def _validate_runtime(self) -> None:
        if self.runtime.sample_rate != self.config.sample_rate:
            raise ValueError(
                "runtime sample_rate must match config sample_rate "
                f"{self.config.sample_rate}, got {self.runtime.sample_rate}"
            )
        if self.runtime.feature_stride_sec <= 0.0:
            raise ValueError("runtime feature_stride_sec must be positive")
        if self.runtime.encoder_subsampling_factor <= 0:
            raise ValueError("runtime encoder_subsampling_factor must be positive")

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

    @staticmethod
    def _load_runtime(config: ParakeetStreamConfig) -> ParakeetRnntRuntime:
        from fa_asr_py.backends.nemo_parakeet_runtime import NemoParakeetRnntRuntime

        return NemoParakeetRnntRuntime(config)
