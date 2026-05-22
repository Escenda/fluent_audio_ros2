import pytest

from fa_audio_mcp.config import load_server_config
from fa_audio_mcp.errors import AudioToolError


_ENV_NAMES = (
    "FLUENT_AUDIO_MCP_TRANSPORT",
    "FLUENT_AUDIO_MCP_HOST",
    "FLUENT_AUDIO_MCP_PORT",
    "FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC",
    "FLUENT_AUDIO_EXPORT_AUDIO_WINDOW_SERVICE",
    "FLUENT_AUDIO_EXPORT_SCOPE_MIC",
    "FLUENT_AUDIO_EXPORT_SCOPE_SYSTEM",
    "FLUENT_AUDIO_EXPORT_SCOPE_MIXED",
    "FLUENT_AUDIO_EXPORT_DEFAULT_SCOPE",
    "FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE",
    "FLUENT_AUDIO_ARCHIVE_SCOPE_MIC",
    "FLUENT_AUDIO_ARCHIVE_SCOPE_SYSTEM",
    "FLUENT_AUDIO_ARCHIVE_SCOPE_MIXED",
    "FLUENT_AUDIO_ARCHIVE_DEFAULT_SCOPE",
    "FLUENT_AUDIO_TIME_MARKERS",
)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_config_rejects_invalid_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FLUENT_AUDIO_MCP_TRANSPORT", "websocket")

    with pytest.raises(AudioToolError) as exc_info:
        load_server_config()

    assert exc_info.value.error_code == "invalid_config"


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_config_rejects_invalid_port(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FLUENT_AUDIO_MCP_PORT", value)

    with pytest.raises(AudioToolError) as exc_info:
        load_server_config()

    assert exc_info.value.error_code == "invalid_config"


@pytest.mark.parametrize("value", ["0", "-0.5", "not-a-float"])
def test_config_rejects_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC", value)

    with pytest.raises(AudioToolError) as exc_info:
        load_server_config()

    assert exc_info.value.error_code == "invalid_config"


@pytest.mark.parametrize(
    "env_name",
    [
        "FLUENT_AUDIO_EXPORT_AUDIO_WINDOW_SERVICE",
        "FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE",
    ],
)
def test_config_rejects_empty_service_name(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv(env_name, " ")

    with pytest.raises(AudioToolError) as exc_info:
        load_server_config()

    assert exc_info.value.error_code == "invalid_config"


def test_config_loads_export_and_archive_scope_configs_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FLUENT_AUDIO_EXPORT_SCOPE_MIC", "mic_export")
    monkeypatch.setenv("FLUENT_AUDIO_EXPORT_SCOPE_SYSTEM", "system_export")
    monkeypatch.setenv("FLUENT_AUDIO_ARCHIVE_SCOPE_MIC", "mic")
    monkeypatch.setenv("FLUENT_AUDIO_ARCHIVE_SCOPE_SYSTEM", "system_archive")

    config = load_server_config()

    assert config.export_scope_config.mic == "mic_export"
    assert config.export_scope_config.system == "system_export"
    assert config.export_scope_config.mixed is None
    assert config.export_scope_config.default_scope_key == "mic"
    assert config.archive_scope_config.mic == "mic"
    assert config.archive_scope_config.system == "system_archive"
    assert config.archive_scope_config.mixed is None
    assert config.archive_scope_config.default_scope_key == "mic"


def test_config_loads_explicit_tool_default_scope_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FLUENT_AUDIO_EXPORT_DEFAULT_SCOPE", "system")
    monkeypatch.setenv("FLUENT_AUDIO_ARCHIVE_DEFAULT_SCOPE", "system")

    config = load_server_config()

    assert config.export_scope_config.default_scope_key == "system"
    assert config.archive_scope_config.default_scope_key == "system"


@pytest.mark.parametrize(
    "env_name",
    [
        "FLUENT_AUDIO_EXPORT_DEFAULT_SCOPE",
        "FLUENT_AUDIO_ARCHIVE_DEFAULT_SCOPE",
    ],
)
def test_config_rejects_invalid_default_scope_key(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv(env_name, "camera")

    with pytest.raises(AudioToolError) as exc_info:
        load_server_config()

    assert exc_info.value.error_code == "invalid_config"


def test_config_loads_time_marker_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv(
        "FLUENT_AUDIO_TIME_MARKERS",
        "action_12.start=1700000000000000000;action_12.end=1700000005000000000",
    )

    config = load_server_config()

    assert (
        config.time_marker_resolver.resolve_endpoint("action_12.start")
        == 1_700_000_000_000_000_000
    )
    assert (
        config.time_marker_resolver.resolve_endpoint("action_12.end+2s")
        == 1_700_000_007_000_000_000
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "action_12.start=",
        "action_12.start=-1",
        "action_12.start=1.5",
        "action_12.middle=1",
        "action_12.start=1;action_12.start=2",
        "action_12.start=1;",
        "action_12.start",
    ],
)
def test_config_rejects_invalid_time_markers(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FLUENT_AUDIO_TIME_MARKERS", value)

    with pytest.raises(AudioToolError) as exc_info:
        load_server_config()

    assert exc_info.value.error_code == "invalid_config"
