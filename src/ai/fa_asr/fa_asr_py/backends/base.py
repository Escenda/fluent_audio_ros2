from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

ASR_AUDIO_ENCODING_FLOAT32LE = "FLOAT32LE"
ASR_AUDIO_ENCODING_PCM16LE = "PCM16LE"
RESULT_FORMAT_PLAIN_TEXT = "plain_text"
RESULT_FORMAT_SEGMENTS_JSON_V1 = "segments_json_v1"
SUPPORTED_RESULT_FORMATS = frozenset(
    (RESULT_FORMAT_PLAIN_TEXT, RESULT_FORMAT_SEGMENTS_JSON_V1)
)


@dataclass(frozen=True)
class AsrBackendCapability:
    audio_encoding: str
    sample_rate_hz: int
    channels: int
    streaming: bool
    final_results_only: bool


@dataclass(frozen=True)
class AsrAudioPayload:
    encoding: str
    sample_rate_hz: int
    channels: int
    data: bytes
    sample_count: int

    @classmethod
    def from_float32_samples(
        cls,
        samples: np.ndarray,
        *,
        sample_rate_hz: int,
        channels: int = 1,
    ) -> "AsrAudioPayload":
        _validate_sample_rate(sample_rate_hz)
        _validate_channels(channels)
        _validate_float32_samples(samples)
        return cls(
            encoding=ASR_AUDIO_ENCODING_FLOAT32LE,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            data=samples.astype("<f4", copy=False).tobytes(),
            sample_count=int(samples.size),
        )

    @classmethod
    def from_pcm16le_bytes(
        cls,
        data: bytes,
        *,
        sample_rate_hz: int,
        channels: int,
    ) -> "AsrAudioPayload":
        _validate_sample_rate(sample_rate_hz)
        _validate_channels(channels)
        if not data:
            raise ValueError("ASR audio payload data is required")
        bytes_per_sample = np.dtype("<i2").itemsize
        frame_width = bytes_per_sample * channels
        if len(data) % frame_width != 0:
            raise ValueError("PCM16LE payload byte length must align to channels")
        return cls(
            encoding=ASR_AUDIO_ENCODING_PCM16LE,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            data=bytes(data),
            sample_count=len(data) // frame_width,
        )

    def float32_samples(self) -> np.ndarray:
        if self.encoding != ASR_AUDIO_ENCODING_FLOAT32LE:
            raise ValueError(f"ASR audio payload encoding must be FLOAT32LE, got {self.encoding}")
        samples = np.frombuffer(self.data, dtype="<f4")
        _validate_float32_samples(samples)
        return samples

    def validate_matches(self, capability: "AsrBackendCapability") -> None:
        if self.encoding != capability.audio_encoding:
            raise ValueError(
                f"ASR request encoding must be {capability.audio_encoding}, got {self.encoding}"
            )
        if self.sample_rate_hz != capability.sample_rate_hz:
            raise ValueError(
                "ASR request sample_rate_hz must be "
                f"{capability.sample_rate_hz}, got {self.sample_rate_hz}"
            )
        if self.channels != capability.channels:
            raise ValueError(
                f"ASR request channels must be {capability.channels}, got {self.channels}"
            )
        if self.sample_count <= 0:
            raise ValueError("ASR request sample_count must be greater than zero")
        if not self.data:
            raise ValueError("ASR request data is required")


@dataclass(frozen=True)
class AsrRequest:
    session_id: str
    user_turn_id: int
    payload: AsrAudioPayload

    @property
    def sample_rate(self) -> int:
        return self.payload.sample_rate_hz

    @property
    def samples(self) -> np.ndarray:
        return self.payload.float32_samples()


@dataclass(frozen=True)
class AsrTranscriptSegment:
    start_sample: int
    end_sample: int
    text: str
    speaker_label: str | None = None


@dataclass(frozen=True)
class AsrTranscript:
    segments: tuple[AsrTranscriptSegment, ...]


@dataclass(frozen=True)
class AsrStreamRequest:
    session_id: str
    user_turn_id: int


@dataclass(frozen=True)
class AsrStreamResult:
    transcript: AsrTranscript
    is_final: bool
    sample_count: int


class AsrStreamingSession(Protocol):
    def push_audio(self, payload: AsrAudioPayload) -> tuple[AsrStreamResult, ...]:
        ...

    def drain_results(self) -> tuple[AsrStreamResult, ...]:
        ...

    def finish(self) -> tuple[AsrStreamResult, ...]:
        ...

    def cancel(self) -> None:
        ...


def validate_result_format(result_format: str) -> str:
    normalized = result_format.strip()
    if not normalized:
        raise RuntimeError("backend.result_format is required")
    if normalized not in SUPPORTED_RESULT_FORMATS:
        raise RuntimeError(f"unsupported backend.result_format: {normalized}")
    return normalized


def plain_text_to_asr_transcript(text: str, *, sample_count: int) -> AsrTranscript:
    transcript = text.strip()
    if not transcript:
        raise RuntimeError("ASR backend returned an empty transcript")
    return build_asr_transcript(
        (
            AsrTranscriptSegment(
                start_sample=0,
                end_sample=sample_count,
                text=transcript,
            ),
        ),
        sample_count=sample_count,
    )


def build_asr_transcript(
    segments: tuple[AsrTranscriptSegment, ...],
    *,
    sample_count: int,
) -> AsrTranscript:
    _validate_sample_count(sample_count)
    if not segments:
        raise RuntimeError("ASR transcript segments must not be empty")

    previous_end_sample = 0
    for segment in segments:
        _validate_segment(segment, sample_count=sample_count)
        if segment.start_sample < previous_end_sample:
            raise RuntimeError("ASR transcript segments must be sorted and non-overlapping")
        previous_end_sample = segment.end_sample
    return AsrTranscript(segments=segments)


def asr_transcript_text(transcript: AsrTranscript) -> str:
    return " ".join(segment.text for segment in transcript.segments).strip()


def _validate_sample_count(sample_count: int) -> None:
    if type(sample_count) is not int:
        raise RuntimeError("ASR transcript sample_count must be an integer")
    if sample_count <= 0:
        raise RuntimeError("ASR transcript sample_count must be greater than zero")


def _validate_segment(segment: AsrTranscriptSegment, *, sample_count: int) -> None:
    if type(segment.start_sample) is not int or type(segment.end_sample) is not int:
        raise RuntimeError("ASR transcript segment sample offsets must be integers")
    if segment.start_sample < 0:
        raise RuntimeError("ASR transcript segment start_sample must be greater than or equal to zero")
    if segment.start_sample >= segment.end_sample:
        raise RuntimeError("ASR transcript segment start_sample must be less than end_sample")
    if segment.end_sample > sample_count:
        raise RuntimeError("ASR transcript segment end_sample exceeds request sample count")
    if type(segment.text) is not str:
        raise RuntimeError("ASR transcript segment text must be a string")
    if not segment.text.strip():
        raise RuntimeError("ASR transcript segment text must not be empty")
    if segment.speaker_label is not None:
        if type(segment.speaker_label) is not str:
            raise RuntimeError("ASR transcript segment speaker_label must be a string when present")
        if not segment.speaker_label.strip():
            raise RuntimeError(
                "ASR transcript segment speaker_label must be a non-empty string when present"
            )


def _validate_sample_rate(sample_rate_hz: int) -> None:
    if type(sample_rate_hz) is not int:
        raise ValueError("ASR audio sample_rate_hz must be an integer")
    if sample_rate_hz <= 0:
        raise ValueError("ASR audio sample_rate_hz must be positive")


def _validate_channels(channels: int) -> None:
    if type(channels) is not int:
        raise ValueError("ASR audio channels must be an integer")
    if channels <= 0:
        raise ValueError("ASR audio channels must be positive")


def _validate_float32_samples(samples: np.ndarray) -> None:
    if samples.dtype != np.float32:
        raise ValueError("ASR request samples must be float32")
    if samples.ndim != 1:
        raise ValueError("ASR request samples must be one-dimensional")
    if samples.size == 0:
        raise ValueError("ASR request samples are required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("ASR request contains non-finite samples")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ValueError("ASR request samples must be normalized to [-1.0, 1.0]")


class AsrBackend(Protocol):
    name: str
    capability: AsrBackendCapability

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        ...


class StreamingAsrBackend(AsrBackend, Protocol):
    def start_stream(self, request: AsrStreamRequest) -> AsrStreamingSession:
        ...
