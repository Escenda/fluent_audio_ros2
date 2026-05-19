from __future__ import annotations

from dataclasses import dataclass
from os import environ

from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.scopes import AudioScopeConfig, AudioScopeKey


@dataclass(frozen=True)
class ServerConfig:
    transport: str
    host: str
    port: int
    export_service_name: str
    archive_service_name: str
    transcribe_service_name: str
    service_timeout_sec: float
    export_scope_config: AudioScopeConfig
    archive_scope_config: AudioScopeConfig
    transcribe_scope_config: AudioScopeConfig


def load_server_config() -> ServerConfig:
    transport = environ.get("FLUENT_AUDIO_MCP_TRANSPORT", "stdio")
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise AudioToolError(
            "invalid_config",
            "FLUENT_AUDIO_MCP_TRANSPORT must be one of stdio, sse, streamable-http",
        )

    return ServerConfig(
        transport=transport,
        host=environ.get("FLUENT_AUDIO_MCP_HOST", "0.0.0.0"),
        port=_read_positive_int("FLUENT_AUDIO_MCP_PORT", "9110"),
        export_service_name=_read_non_empty_string(
            "FLUENT_AUDIO_EXPORT_AUDIO_WINDOW_SERVICE",
            "export_audio_window",
        ),
        archive_service_name=_read_non_empty_string(
            "FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE",
            "archive_audio_window",
        ),
        transcribe_service_name=_read_non_empty_string(
            "FLUENT_AUDIO_TRANSCRIBE_AUDIO_SERVICE",
            "transcribe_audio",
        ),
        service_timeout_sec=_read_positive_float(
            "FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC",
            "10.0",
        ),
        export_scope_config=AudioScopeConfig(
            mic=_read_non_empty_string("FLUENT_AUDIO_EXPORT_SCOPE_MIC", "mic"),
            system=_read_optional_scope("FLUENT_AUDIO_EXPORT_SCOPE_SYSTEM"),
            mixed=_read_optional_scope("FLUENT_AUDIO_EXPORT_SCOPE_MIXED"),
            default_scope_key=_read_optional_scope_key(
                "FLUENT_AUDIO_EXPORT_DEFAULT_SCOPE",
                "mic",
            ),
        ),
        archive_scope_config=AudioScopeConfig(
            mic=_read_non_empty_string("FLUENT_AUDIO_ARCHIVE_SCOPE_MIC", "mic"),
            system=_read_optional_scope("FLUENT_AUDIO_ARCHIVE_SCOPE_SYSTEM"),
            mixed=_read_optional_scope("FLUENT_AUDIO_ARCHIVE_SCOPE_MIXED"),
            default_scope_key=_read_optional_scope_key(
                "FLUENT_AUDIO_ARCHIVE_DEFAULT_SCOPE",
                "mic",
            ),
        ),
        transcribe_scope_config=AudioScopeConfig(
            mic=_read_optional_scope("FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIC"),
            system=_read_optional_scope("FLUENT_AUDIO_TRANSCRIBE_SCOPE_SYSTEM"),
            mixed=_read_optional_scope("FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIXED"),
            default_scope_key=_read_optional_scope_key(
                "FLUENT_AUDIO_TRANSCRIBE_DEFAULT_SCOPE",
                None,
            ),
        ),
    )


def _read_non_empty_string(name: str, default: str) -> str:
    value = environ.get(name, default).strip()
    if value == "":
        raise AudioToolError("invalid_config", f"{name} must be non-empty")
    return value


def _read_optional_scope(name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    stripped_value = value.strip()
    if stripped_value == "":
        return None
    return stripped_value


def _read_optional_scope_key(name: str, default: str | None) -> AudioScopeKey | None:
    raw_value = environ.get(name, default)
    if raw_value is None:
        return None
    value = raw_value.strip()
    if value == "":
        return None
    if value == "mic":
        return "mic"
    if value == "system":
        return "system"
    if value == "mixed":
        return "mixed"
    raise AudioToolError(
        "invalid_config",
        f"{name} must be one of mic, system, mixed",
    )


def _read_positive_float(name: str, default: str) -> float:
    raw_value = environ.get(name, default)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise AudioToolError("invalid_config", f"{name} must be a number") from exc
    if value <= 0.0:
        raise AudioToolError("invalid_config", f"{name} must be greater than 0")
    return value


def _read_positive_int(name: str, default: str) -> int:
    raw_value = environ.get(name, default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise AudioToolError("invalid_config", f"{name} must be an integer") from exc
    if value <= 0:
        raise AudioToolError("invalid_config", f"{name} must be greater than 0")
    return value
