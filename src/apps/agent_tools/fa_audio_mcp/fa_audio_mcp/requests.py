from __future__ import annotations

from dataclasses import dataclass

from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.scopes import AudioScopeResolver
from fa_audio_mcp.time_range import NumericTimeRange, TimeMarkerResolver, resolve_time_range


DEFAULT_AUDIO_CLIP_CODEC = "pcm_s16le"
DEFAULT_AUDIO_CLIP_CONTAINER = "wav"
DEFAULT_AUDIO_CLIP_PAYLOAD_FORMAT = "audio/wav"


@dataclass(frozen=True)
class ExportAudioRequestValues:
    time_range: NumericTimeRange
    time_range_spec: str
    audio_scope: str
    codec: str
    container: str
    payload_format: str


@dataclass(frozen=True)
class ArchiveAudioRequestValues:
    time_range: NumericTimeRange
    time_range_spec: str
    audio_scope: str
    reason: str
    related_artifact_ids: list[str]
    codec: str
    container: str
    payload_format: str


def build_export_audio_request_values(
    *,
    time_range: str,
    audio_scope: str | None,
    scope_resolver: AudioScopeResolver,
    codec: str | None = None,
    container: str | None = None,
    payload_format: str | None = None,
    now_unix_ns: int | None = None,
    marker_resolver: TimeMarkerResolver | None = None,
) -> ExportAudioRequestValues:
    parsed_time_range = resolve_time_range(
        time_range,
        now_unix_ns=now_unix_ns,
        marker_resolver=marker_resolver,
    )
    resolved_scope = scope_resolver.resolve(audio_scope)
    return ExportAudioRequestValues(
        time_range=parsed_time_range,
        time_range_spec=parsed_time_range.spec,
        audio_scope=resolved_scope,
        codec=_optional_string(codec, DEFAULT_AUDIO_CLIP_CODEC),
        container=_optional_string(container, DEFAULT_AUDIO_CLIP_CONTAINER),
        payload_format=_optional_string(payload_format, DEFAULT_AUDIO_CLIP_PAYLOAD_FORMAT),
    )


def build_archive_audio_request_values(
    *,
    time_range: str,
    audio_scope: str | None,
    reason: str,
    related_artifact_ids: list[str],
    scope_resolver: AudioScopeResolver,
    codec: str | None = None,
    container: str | None = None,
    payload_format: str | None = None,
    now_unix_ns: int | None = None,
    marker_resolver: TimeMarkerResolver | None = None,
) -> ArchiveAudioRequestValues:
    normalized_reason = reason.strip()
    if normalized_reason == "":
        raise AudioToolError("invalid_archive_request", "reason must be non-empty")

    parsed_time_range = resolve_time_range(
        time_range,
        now_unix_ns=now_unix_ns,
        marker_resolver=marker_resolver,
    )
    resolved_scope = scope_resolver.resolve(audio_scope)
    return ArchiveAudioRequestValues(
        time_range=parsed_time_range,
        time_range_spec=parsed_time_range.spec,
        audio_scope=resolved_scope,
        reason=normalized_reason,
        related_artifact_ids=list(related_artifact_ids),
        codec=_optional_string(codec, DEFAULT_AUDIO_CLIP_CODEC),
        container=_optional_string(container, DEFAULT_AUDIO_CLIP_CONTAINER),
        payload_format=_optional_string(payload_format, DEFAULT_AUDIO_CLIP_PAYLOAD_FORMAT),
    )


def _optional_string(value: str | None, default: str) -> str:
    if value is None:
        return default
    return value
