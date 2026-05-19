from __future__ import annotations

from typing import Protocol

from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.json_types import JsonValue
from fa_audio_mcp.time_range import NumericTimeRange, requested_time_range_spec


class TimeRangeLike(Protocol):
    start_unix_ns: int
    end_unix_ns: int
    clock: str
    uncertainty_ns: int
    uncertainty_reason: str


class AudioClipRefLike(Protocol):
    clip_id: str
    uri: str
    codec: str
    container: str
    payload_format: str
    sample_rate: int
    channels: int
    duration_ns: int
    time_range: TimeRangeLike


class AudioClipResponseLike(Protocol):
    success: bool
    error_code: str
    message: str
    audio_clip_ref: AudioClipRefLike
    time_range: TimeRangeLike


class TranscriptSegmentLike(Protocol):
    start_unix_ns: int
    end_unix_ns: int
    text: str
    speaker_label: str


class AudioWindowRefLike(Protocol):
    window_id: str
    window_epoch: int
    source_id: str
    stream_id: str
    time_range: TimeRangeLike


class AudioModelRefLike(Protocol):
    backend_name: str
    backend_kind: str
    model_id: str
    model_path: str
    model_version: str
    model_revision: str


class TranscribeAudioResponseLike(Protocol):
    success: bool
    error_code: str
    message: str
    segments: list[TranscriptSegmentLike]
    audio_window_ref: AudioWindowRefLike
    model_ref: AudioModelRefLike
    time_range: TimeRangeLike


def format_export_audio_result(
    response: AudioClipResponseLike,
    requested_time_range: NumericTimeRange,
) -> dict[str, JsonValue]:
    return _format_audio_clip_result(response, requested_time_range)


def format_archive_audio_result(
    response: AudioClipResponseLike,
    requested_time_range: NumericTimeRange,
) -> dict[str, JsonValue]:
    return _format_audio_clip_result(response, requested_time_range)


def _format_audio_clip_result(
    response: AudioClipResponseLike,
    requested_time_range: NumericTimeRange,
) -> dict[str, JsonValue]:
    if not response.success:
        raise AudioToolError(response.error_code, response.message)

    clip_ref = response.audio_clip_ref
    return {
        "audio_clip_ref": {
            "clip_id": clip_ref.clip_id,
            "uri": clip_ref.uri,
            "codec": clip_ref.codec,
            "container": clip_ref.container,
            "payload_format": clip_ref.payload_format,
            "sample_rate": clip_ref.sample_rate,
            "channels": clip_ref.channels,
            "duration_ns": clip_ref.duration_ns,
            "time_range": _format_time_range(clip_ref.time_range),
        },
        "time_range": _format_time_range(response.time_range),
        "requested_time_range": _format_requested_time_range(requested_time_range),
    }


def format_transcribe_audio_result(
    response: TranscribeAudioResponseLike,
    requested_time_range: NumericTimeRange,
) -> dict[str, JsonValue]:
    if not response.success:
        raise AudioToolError(response.error_code, response.message)

    return {
        "segments": [_format_segment(segment) for segment in response.segments],
        "audio_window_ref": _format_audio_window_ref(response.audio_window_ref),
        "model_ref": _format_model_ref(response.model_ref),
        "time_range": _format_time_range(response.time_range),
        "requested_time_range": _format_requested_time_range(requested_time_range),
    }


def _format_time_range(time_range: TimeRangeLike) -> dict[str, JsonValue]:
    return {
        "start_unix_ns": time_range.start_unix_ns,
        "end_unix_ns": time_range.end_unix_ns,
        "clock": time_range.clock,
        "uncertainty_ns": time_range.uncertainty_ns,
        "uncertainty_reason": time_range.uncertainty_reason,
    }


def _format_requested_time_range(time_range: NumericTimeRange) -> dict[str, JsonValue]:
    return {
        "start_unix_ns": time_range.start_unix_ns,
        "end_unix_ns": time_range.end_unix_ns,
        "spec": requested_time_range_spec(time_range),
    }


def _format_segment(segment: TranscriptSegmentLike) -> dict[str, JsonValue]:
    return {
        "start_unix_ns": segment.start_unix_ns,
        "end_unix_ns": segment.end_unix_ns,
        "text": segment.text,
        "speaker_label": segment.speaker_label,
    }


def _format_audio_window_ref(ref: AudioWindowRefLike) -> dict[str, JsonValue]:
    return {
        "window_id": ref.window_id,
        "window_epoch": ref.window_epoch,
        "source_id": ref.source_id,
        "stream_id": ref.stream_id,
        "time_range": _format_time_range(ref.time_range),
    }


def _format_model_ref(ref: AudioModelRefLike) -> dict[str, JsonValue]:
    return {
        "backend_name": ref.backend_name,
        "backend_kind": ref.backend_kind,
        "model_id": ref.model_id,
        "model_path": ref.model_path,
        "model_version": ref.model_version,
        "model_revision": ref.model_revision,
    }
