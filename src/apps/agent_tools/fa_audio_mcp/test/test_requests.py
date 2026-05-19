import pytest

from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.requests import (
    build_archive_audio_request_values,
    build_transcribe_audio_request_values,
)
from fa_audio_mcp.scopes import AudioScopeConfig, AudioScopeResolver


def test_archive_request_validation_requires_reason() -> None:
    resolver = AudioScopeResolver(AudioScopeConfig(mic="mic"))

    with pytest.raises(AudioToolError):
        build_archive_audio_request_values(
            time_range="1..2",
            audio_scope="mic",
            reason=" ",
            related_artifact_ids=[],
            scope_resolver=resolver,
        )


def test_archive_request_validation_produces_deterministic_values() -> None:
    resolver = AudioScopeResolver(AudioScopeConfig(mic="robot_mic"))

    values = build_archive_audio_request_values(
        time_range="100..300",
        audio_scope="mic",
        reason=" investigation evidence ",
        related_artifact_ids=["artifact-1", "artifact-2"],
        scope_resolver=resolver,
        codec="flac",
        container="wav",
        payload_format="file",
    )

    assert values.time_range_spec == "100..300"
    assert values.time_range.start_unix_ns == 100
    assert values.time_range.end_unix_ns == 300
    assert values.audio_scope == "robot_mic"
    assert values.reason == "investigation evidence"
    assert values.related_artifact_ids == ["artifact-1", "artifact-2"]
    assert values.codec == "flac"
    assert values.container == "wav"
    assert values.payload_format == "file"


def test_archive_request_uses_supported_format_defaults_when_omitted() -> None:
    resolver = AudioScopeResolver(AudioScopeConfig(mic="mic"))

    values = build_archive_audio_request_values(
        time_range="100..300",
        audio_scope="mic",
        reason="incident evidence",
        related_artifact_ids=[],
        scope_resolver=resolver,
    )

    assert values.codec == "pcm_s16le"
    assert values.container == "wav"
    assert values.payload_format == "audio/wav"


@pytest.mark.parametrize("scope", [None, " "])
def test_archive_request_uses_configured_default_scope_when_omitted(
    scope: str | None,
) -> None:
    resolver = AudioScopeResolver(
        AudioScopeConfig(
            mic="robot_mic",
            default_scope_key="mic",
        )
    )

    values = build_archive_audio_request_values(
        time_range="100..300",
        audio_scope=scope,
        reason="incident evidence",
        related_artifact_ids=[],
        scope_resolver=resolver,
    )

    assert values.audio_scope == "robot_mic"


def test_transcribe_request_rejects_unconfigured_mic_scope() -> None:
    resolver = AudioScopeResolver(AudioScopeConfig())

    with pytest.raises(AudioToolError):
        build_transcribe_audio_request_values(
            time_range="100..300",
            audio_scope="mic",
            scope_resolver=resolver,
        )


def test_transcribe_request_uses_configured_asr_stream_scope() -> None:
    resolver = AudioScopeResolver(AudioScopeConfig(mic="audio/high_pass/mic"))

    values = build_transcribe_audio_request_values(
        time_range="100..300",
        audio_scope="mic",
        scope_resolver=resolver,
    )

    assert values.audio_scope == "audio/high_pass/mic"


def test_transcribe_request_uses_configured_default_scope_when_omitted() -> None:
    resolver = AudioScopeResolver(
        AudioScopeConfig(
            mic="audio/high_pass/mic",
            default_scope_key="mic",
        )
    )

    values = build_transcribe_audio_request_values(
        time_range="100..300",
        audio_scope=None,
        scope_resolver=resolver,
    )

    assert values.audio_scope == "audio/high_pass/mic"
