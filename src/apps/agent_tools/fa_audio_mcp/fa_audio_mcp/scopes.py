from __future__ import annotations

from dataclasses import dataclass

from fa_audio_mcp.errors import AudioToolError


@dataclass(frozen=True)
class AudioScopeConfig:
    mic: str | None = None
    system: str | None = None
    mixed: str | None = None


class AudioScopeResolver:
    def __init__(self, config: AudioScopeConfig) -> None:
        self._config = config

    def resolve(self, audio_scope: str) -> str:
        requested_scope = audio_scope.strip()
        if requested_scope == "":
            raise AudioToolError("unsupported_audio_scope", "audio_scope must be non-empty")
        if requested_scope == "mic" and self._config.mic is not None:
            return self._config.mic
        if requested_scope == "system" and self._config.system is not None:
            return self._config.system
        if requested_scope == "mixed" and self._config.mixed is not None:
            return self._config.mixed
        raise AudioToolError(
            "unsupported_audio_scope",
            f"audio_scope '{requested_scope}' is not configured",
        )
