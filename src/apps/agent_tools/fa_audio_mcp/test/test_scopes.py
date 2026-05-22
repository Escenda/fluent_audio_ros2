import pytest

from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.config import load_server_config
from fa_audio_mcp.scopes import AudioScopeConfig, AudioScopeResolver


_ENV_NAMES = (
    "FLUENT_AUDIO_MCP_TRANSPORT",
    "FLUENT_AUDIO_MCP_PORT",
    "FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC",
    "FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE",
    "FLUENT_AUDIO_ARCHIVE_SCOPE_MIC",
    "FLUENT_AUDIO_ARCHIVE_SCOPE_SYSTEM",
    "FLUENT_AUDIO_ARCHIVE_SCOPE_MIXED",
    "FLUENT_AUDIO_ARCHIVE_DEFAULT_SCOPE",
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


@pytest.mark.parametrize("scope", [None, " ", "\t"])
def test_archive_omitted_scope_resolves_to_explicit_default_config(
    monkeypatch: pytest.MonkeyPatch,
    scope: str | None,
) -> None:
    _clear_env(monkeypatch)

    config = load_server_config()
    resolver = AudioScopeResolver(config.archive_scope_config)

    assert resolver.resolve(scope) == "mic"


def test_archive_default_scope_fails_when_target_scope_is_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FLUENT_AUDIO_ARCHIVE_DEFAULT_SCOPE", "system")

    config = load_server_config()
    resolver = AudioScopeResolver(config.archive_scope_config)

    with pytest.raises(AudioToolError) as exc_info:
        resolver.resolve(None)

    assert exc_info.value.error_code == "unsupported_audio_scope"
