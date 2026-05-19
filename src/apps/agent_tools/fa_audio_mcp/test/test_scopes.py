import pytest

from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.config import load_server_config
from fa_audio_mcp.scopes import AudioScopeConfig, AudioScopeResolver


_ENV_NAMES = (
    "FLUENT_AUDIO_MCP_TRANSPORT",
    "FLUENT_AUDIO_MCP_PORT",
    "FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC",
    "FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE",
    "FLUENT_AUDIO_TRANSCRIBE_AUDIO_SERVICE",
    "FLUENT_AUDIO_ARCHIVE_SCOPE_MIC",
    "FLUENT_AUDIO_ARCHIVE_SCOPE_SYSTEM",
    "FLUENT_AUDIO_ARCHIVE_SCOPE_MIXED",
    "FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIC",
    "FLUENT_AUDIO_TRANSCRIBE_SCOPE_SYSTEM",
    "FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIXED",
)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_scope_resolver_maps_configured_scopes() -> None:
    resolver = AudioScopeResolver(
        AudioScopeConfig(
            mic="robot_mic",
            system="system_bus",
            mixed="agent_mix",
        )
    )

    assert resolver.resolve("mic") == "robot_mic"
    assert resolver.resolve("system") == "system_bus"
    assert resolver.resolve("mixed") == "agent_mix"


@pytest.mark.parametrize("scope", ["system", "mixed", "camera", ""])
def test_scope_resolver_rejects_unsupported_and_unknown_scopes(scope: str) -> None:
    resolver = AudioScopeResolver(AudioScopeConfig(mic="mic"))

    with pytest.raises(AudioToolError):
        resolver.resolve(scope)


def test_archive_mic_resolves_with_default_archive_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)

    config = load_server_config()
    resolver = AudioScopeResolver(config.archive_scope_config)

    assert resolver.resolve("mic") == "mic"


def test_transcribe_mic_fails_without_configured_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)

    config = load_server_config()
    resolver = AudioScopeResolver(config.transcribe_scope_config)

    with pytest.raises(AudioToolError):
        resolver.resolve("mic")


def test_transcribe_mic_resolves_to_configured_asr_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIC", "audio/high_pass/mic")

    config = load_server_config()
    resolver = AudioScopeResolver(config.transcribe_scope_config)

    assert resolver.resolve("mic") == "audio/high_pass/mic"
