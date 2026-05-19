from dataclasses import dataclass

import pytest

from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.responses import (
    format_archive_audio_result,
    format_export_audio_result,
    format_transcribe_audio_result,
)
from fa_audio_mcp.time_range import NumericTimeRange


@dataclass
class FakeTimeRange:
    start_unix_ns: int
    end_unix_ns: int
    clock: str = "robot"
    uncertainty_ns: int = 0
    uncertainty_reason: str = ""


@dataclass
class FakeAudioClipRef:
    clip_id: str
    uri: str
    metadata_uri: str
    content_sha256: str
    metadata_sha256: str
    codec: str
    container: str
    payload_format: str
    sample_rate: int
    channels: int
    duration_ns: int
    time_range: FakeTimeRange


@dataclass
class FakeArchiveAudioResponse:
    success: bool
    error_code: str
    message: str
    audio_clip_ref: FakeAudioClipRef
    time_range: FakeTimeRange


@dataclass
class FakeTranscriptSegment:
    start_unix_ns: int
    end_unix_ns: int
    text: str
    speaker_label: str


@dataclass
class FakeAudioWindowRef:
    window_id: str
    window_epoch: int
    source_id: str
    stream_id: str
    time_range: FakeTimeRange


@dataclass
class FakeAudioModelRef:
    backend_name: str
    backend_kind: str
    model_id: str
    model_path: str
    model_version: str
    model_revision: str


@dataclass
class FakeTranscribeAudioResponse:
    success: bool
    error_code: str
    message: str
    segments: list[FakeTranscriptSegment]
    audio_window_ref: FakeAudioWindowRef
    model_ref: FakeAudioModelRef
    time_range: FakeTimeRange


def test_archive_response_formatter_raises_tool_error_on_failure() -> None:
    response = _archive_response(success=False, error_code="range_outside_window")

    with pytest.raises(AudioToolError) as exc_info:
        format_archive_audio_result(response, NumericTimeRange(10, 20))

    assert exc_info.value.error_code == "range_outside_window"
    assert exc_info.value.message == "range_outside_window message"


def test_export_response_formatter_raises_tool_error_on_failure() -> None:
    response = _archive_response(success=False, error_code="export_failed")

    with pytest.raises(AudioToolError) as exc_info:
        format_export_audio_result(response, NumericTimeRange(10, 20))

    assert exc_info.value.error_code == "export_failed"
    assert exc_info.value.message == "export_failed message"


def test_archive_response_formatter_returns_clip_and_time_range_data() -> None:
    response = _archive_response(success=True, error_code="none")

    result = format_archive_audio_result(response, NumericTimeRange(10, 20))

    assert result["audio_clip_ref"] == {
        "clip_id": "clip-1",
        "uri": "s3://daihen/v2/audio/clip-1.flac",
        "metadata_uri": "s3://daihen/v2/audio/clip-1.metadata.json",
        "content_sha256": "0" * 64,
        "metadata_sha256": "1" * 64,
        "codec": "flac",
        "container": "wav",
        "payload_format": "file",
        "sample_rate": 16000,
        "channels": 1,
        "duration_ns": 10,
        "time_range": {
            "start_unix_ns": 10,
            "end_unix_ns": 20,
            "clock": "robot",
            "uncertainty_ns": 0,
            "uncertainty_reason": "",
        },
    }
    assert result["time_range"] == {
        "start_unix_ns": 10,
        "end_unix_ns": 20,
        "clock": "robot",
        "uncertainty_ns": 0,
        "uncertainty_reason": "",
    }
    assert result["requested_time_range"] == {
        "start_unix_ns": 10,
        "end_unix_ns": 20,
        "spec": "10..20",
    }


def test_export_response_formatter_returns_clip_and_time_range_data() -> None:
    response = _archive_response(success=True, error_code="none")

    result = format_export_audio_result(response, NumericTimeRange(10, 20))

    assert result["audio_clip_ref"]["clip_id"] == "clip-1"
    assert result["audio_clip_ref"]["uri"] == "s3://daihen/v2/audio/clip-1.flac"
    assert result["audio_clip_ref"]["metadata_uri"] == "s3://daihen/v2/audio/clip-1.metadata.json"
    assert result["audio_clip_ref"]["content_sha256"] == "0" * 64
    assert result["audio_clip_ref"]["metadata_sha256"] == "1" * 64
    assert result["time_range"] == {
        "start_unix_ns": 10,
        "end_unix_ns": 20,
        "clock": "robot",
        "uncertainty_ns": 0,
        "uncertainty_reason": "",
    }
    assert result["requested_time_range"] == {
        "start_unix_ns": 10,
        "end_unix_ns": 20,
        "spec": "10..20",
    }


def test_transcribe_response_formatter_raises_tool_error_on_failure() -> None:
    response = _transcribe_response(success=False, error_code="transcribe_failed")

    with pytest.raises(AudioToolError) as exc_info:
        format_transcribe_audio_result(response, NumericTimeRange(10, 20))

    assert exc_info.value.error_code == "transcribe_failed"
    assert exc_info.value.message == "transcribe_failed message"


def test_transcribe_response_formatter_returns_segments_model_window_and_time_range_data() -> None:
    response = _transcribe_response(success=True, error_code="none")

    result = format_transcribe_audio_result(response, NumericTimeRange(10, 20))

    assert result["segments"] == [
        {
            "start_unix_ns": 10,
            "end_unix_ns": 20,
            "text": "hello",
            "speaker_label": "speaker-1",
        }
    ]
    assert result["audio_window_ref"] == {
        "window_id": "window-1",
        "window_epoch": 7,
        "source_id": "mic",
        "stream_id": "stream-1",
        "time_range": {
            "start_unix_ns": 10,
            "end_unix_ns": 20,
            "clock": "robot",
            "uncertainty_ns": 0,
            "uncertainty_reason": "",
        },
    }
    assert result["model_ref"] == {
        "backend_name": "local",
        "backend_kind": "asr",
        "model_id": "model-1",
        "model_path": "/models/model-1",
        "model_version": "1",
        "model_revision": "rev-1",
    }
    assert result["time_range"] == {
        "start_unix_ns": 10,
        "end_unix_ns": 20,
        "clock": "robot",
        "uncertainty_ns": 0,
        "uncertainty_reason": "",
    }
    assert result["requested_time_range"] == {
        "start_unix_ns": 10,
        "end_unix_ns": 20,
        "spec": "10..20",
    }


def _archive_response(success: bool, error_code: str) -> FakeArchiveAudioResponse:
    time_range = FakeTimeRange(start_unix_ns=10, end_unix_ns=20)
    return FakeArchiveAudioResponse(
        success=success,
        error_code=error_code,
        message=f"{error_code} message",
        audio_clip_ref=FakeAudioClipRef(
            clip_id="clip-1",
            uri="s3://daihen/v2/audio/clip-1.flac",
            metadata_uri="s3://daihen/v2/audio/clip-1.metadata.json",
            content_sha256="0" * 64,
            metadata_sha256="1" * 64,
            codec="flac",
            container="wav",
            payload_format="file",
            sample_rate=16000,
            channels=1,
            duration_ns=10,
            time_range=time_range,
        ),
        time_range=time_range,
    )


def _transcribe_response(success: bool, error_code: str) -> FakeTranscribeAudioResponse:
    time_range = FakeTimeRange(start_unix_ns=10, end_unix_ns=20)
    return FakeTranscribeAudioResponse(
        success=success,
        error_code=error_code,
        message=f"{error_code} message",
        segments=[
            FakeTranscriptSegment(
                start_unix_ns=10,
                end_unix_ns=20,
                text="hello",
                speaker_label="speaker-1",
            )
        ],
        audio_window_ref=FakeAudioWindowRef(
            window_id="window-1",
            window_epoch=7,
            source_id="mic",
            stream_id="stream-1",
            time_range=time_range,
        ),
        model_ref=FakeAudioModelRef(
            backend_name="local",
            backend_kind="asr",
            model_id="model-1",
            model_path="/models/model-1",
            model_version="1",
            model_revision="rev-1",
        ),
        time_range=time_range,
    )
