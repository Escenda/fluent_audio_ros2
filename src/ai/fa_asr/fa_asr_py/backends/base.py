from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

RESULT_FORMAT_PLAIN_TEXT = "plain_text"
RESULT_FORMAT_SEGMENTS_JSON_V1 = "segments_json_v1"
SUPPORTED_RESULT_FORMATS = frozenset(
    (RESULT_FORMAT_PLAIN_TEXT, RESULT_FORMAT_SEGMENTS_JSON_V1)
)


@dataclass(frozen=True)
class AsrRequest:
    session_id: str
    user_turn_id: int
    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class AsrTranscriptSegment:
    start_sample: int
    end_sample: int
    text: str
    speaker_label: str | None = None


@dataclass(frozen=True)
class AsrTranscript:
    segments: tuple[AsrTranscriptSegment, ...]


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


class AsrBackend(Protocol):
    name: str

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        ...
