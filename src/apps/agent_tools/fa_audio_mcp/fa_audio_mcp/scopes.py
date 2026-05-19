from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fa_audio_mcp.errors import AudioToolError

AudioScopeKey = Literal["mic", "system", "mixed"]


@dataclass(frozen=True)
class AudioScopeConfig:
    mic: str | None = None
    system: str | None = None
    mixed: str | None = None
    default_scope_key: AudioScopeKey | None = None


class AudioScopeResolver:
    def __init__(self, config: AudioScopeConfig) -> None:
        self._config = config

    def resolve(self, audio_scope: str | None) -> str:
        requested_scope = self._normalize_requested_scope(audio_scope)
        configured_scope = self._configured_scope(requested_scope)
        if configured_scope is not None:
            return configured_scope
        raise AudioToolError(
            "unsupported_audio_scope",
            f"audio_scope '{requested_scope}' is not configured",
        )

    def _normalize_requested_scope(self, audio_scope: str | None) -> AudioScopeKey:
        if audio_scope is None:
            return self._resolve_default_scope_key()

        requested_scope = audio_scope.strip()
        if requested_scope == "":
            return self._resolve_default_scope_key()
        if requested_scope == "mic":
            return "mic"
        if requested_scope == "system":
            return "system"
        if requested_scope == "mixed":
            return "mixed"
        raise AudioToolError(
            "unsupported_audio_scope",
            f"audio_scope '{requested_scope}' is not configured",
        )

    def _resolve_default_scope_key(self) -> AudioScopeKey:
        default_scope_key = self._config.default_scope_key
        if default_scope_key is None:
            raise AudioToolError(
                "unsupported_audio_scope",
                "audio_scope default is not configured",
            )
        return default_scope_key

    def _configured_scope(self, scope_key: AudioScopeKey) -> str | None:
        if scope_key == "mic":
            return self._config.mic
        if scope_key == "system":
            return self._config.system
        return self._config.mixed
