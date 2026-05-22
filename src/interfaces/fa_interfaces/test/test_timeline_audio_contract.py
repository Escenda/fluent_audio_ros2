from fa_interfaces.msg import (
    AudioClipRef,
    AudioWindowRef,
    ResolvedTimeRange,
)
from fa_interfaces.srv import ArchiveAudioWindow, ExportAudioWindow


def test_resolved_time_range_tracks_clock_and_uncertainty() -> None:
    time_range = ResolvedTimeRange(
        start_unix_ns=1779120000000000000,
        end_unix_ns=1779120005000000000,
        clock=ResolvedTimeRange.CLOCK_MEDIA,
        uncertainty_ns=2500000,
        uncertainty_reason="media_clock_drift_estimate",
    )

    assert time_range.clock == ResolvedTimeRange.CLOCK_MEDIA
    assert time_range.uncertainty_ns == 2500000
    assert time_range.uncertainty_reason == "media_clock_drift_estimate"


def test_audio_refs_can_describe_window_and_clip_contracts() -> None:
    time_range = ResolvedTimeRange(
        start_unix_ns=1779120000000000000,
        end_unix_ns=1779120010000000000,
        clock=ResolvedTimeRange.CLOCK_AGENT,
        uncertainty_ns=0,
        uncertainty_reason="",
    )
    window_ref = AudioWindowRef(
        window_id="fa_audio_window.default",
        window_epoch=7,
        source_id="mic_front",
        stream_id="processed_mic",
        time_range=time_range,
    )
    clip_ref = AudioClipRef(
        clip_id="clip_20260519_001",
        uri="s3://daihen/v2/audio/clip_20260519_001.wav",
        metadata_uri="s3://daihen/v2/audio/clip_20260519_001.metadata.json",
        content_sha256="0" * 64,
        metadata_sha256="1" * 64,
        codec="pcm_s16le",
        container="wav",
        payload_format="audio/wav",
        sample_rate=16000,
        channels=1,
        duration_ns=10000000000,
        time_range=time_range,
    )

    assert window_ref.time_range.clock == ResolvedTimeRange.CLOCK_AGENT
    assert clip_ref.uri.startswith("s3://")
    assert clip_ref.metadata_uri.endswith(".metadata.json")
    assert len(clip_ref.content_sha256) == 64
    assert len(clip_ref.metadata_sha256) == 64
    assert clip_ref.sample_rate == 16000


def test_export_and_archive_return_explicit_clip_or_error_contracts() -> None:
    export_request = ExportAudioWindow.Request(
        time_range_spec="action_12.start..action_12.end+2s",
        audio_scope="mic",
        codec="pcm_s16le",
        container="wav",
        payload_format="audio/wav",
    )
    archive_request = ArchiveAudioWindow.Request(
        time_range_spec="action_12.start..action_12.end+2s",
        audio_scope="mic",
        reason="operator stop instruction evidence",
        related_artifact_ids=["action_12"],
        codec="pcm_s16le",
        container="wav",
        payload_format="audio/wav",
    )
    failure = ArchiveAudioWindow.Response(
        success=False,
        error_code=ArchiveAudioWindow.Response.ERROR_RANGE_OUTSIDE_WINDOW,
        message="requested time range is outside the retained audio window",
    )
    invalid_archive_request = ArchiveAudioWindow.Response(
        success=False,
        error_code=ArchiveAudioWindow.Response.ERROR_INVALID_ARCHIVE_REQUEST,
        message="archive reason is required",
    )
    gap_failure = ArchiveAudioWindow.Response(
        success=False,
        error_code=ArchiveAudioWindow.Response.ERROR_RANGE_NOT_CONTINUOUS,
        message="requested time range is not continuously covered by retained audio",
    )

    assert export_request.payload_format == "audio/wav"
    assert archive_request.related_artifact_ids == ["action_12"]
    assert failure.success is False
    assert failure.error_code == ArchiveAudioWindow.Response.ERROR_RANGE_OUTSIDE_WINDOW
    assert invalid_archive_request.error_code == ArchiveAudioWindow.Response.ERROR_INVALID_ARCHIVE_REQUEST
    assert gap_failure.error_code == ArchiveAudioWindow.Response.ERROR_RANGE_NOT_CONTINUOUS
