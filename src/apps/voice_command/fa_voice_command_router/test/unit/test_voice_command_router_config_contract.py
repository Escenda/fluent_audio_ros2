import re
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def read_package_file(relative_path: str) -> str:
    return (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")


def test_default_config_provides_all_required_startup_parameters() -> None:
    config = yaml.safe_load(read_package_file("config/default.yaml"))
    params = config["fa_voice_command_router"]["ros__parameters"]

    assert params["command_topic"] == "voice/command"
    assert params["state_topic"] == "voice/router/state"
    assert params["active"] is False
    assert params["mode"] == "standby"
    assert params["allowed_modes"] == ["standby", "command", "dictation", "mute"]
    assert params["announce_tts"] is False
    assert params["tts_service"] == "speak"
    assert params["tts_voice_id"] == ""
    assert params["stop_output_on_stop"] is True
    assert params["output_stop_topic"] == "audio/output/stop"


def test_router_declares_required_parameters_without_runtime_defaults() -> None:
    source = read_package_file("fa_voice_command_router_py/router_node.py")

    assert "self.declare_parameter(parameter_name, parameter_type)" in source
    assert "Parameter.Type.STRING" in source
    assert "Parameter.Type.BOOL" in source
    assert "Parameter.Type.STRING_ARRAY" in source
    assert 'self.declare_parameter("command_topic",' not in source
    assert 'self.declare_parameter("state_topic",' not in source
    assert 'self.declare_parameter("active",' not in source
    assert 'self.declare_parameter("mode",' not in source
    assert 'self.declare_parameter("allowed_modes",' not in source
    assert 'self.declare_parameter("announce_tts",' not in source
    assert 'self.declare_parameter("tts_service",' not in source
    assert 'self.declare_parameter("tts_voice_id",' not in source
    assert 'self.declare_parameter("stop_output_on_stop",' not in source
    assert 'self.declare_parameter("output_stop_topic",' not in source


def test_router_uses_typed_required_parameter_readers_instead_of_casts() -> None:
    source = read_package_file("fa_voice_command_router_py/router_node.py")

    assert "def _read_required_string(" in source
    assert "def _read_required_bool(" in source
    assert "def _read_required_string_array(" in source
    assert re.search(r"\bbool\s*\(", source) is None
    assert re.search(r"\bstr\s*\(", source) is None
    forbidden_type_escapes = (
        "dict[str, " + "A" + "ny]",
        "Dict[str, " + "A" + "ny]",
        "A" + "ny",
        "o" + "bject",
        "# type: " + "ignore",
        "ca" + "st(",
    )
    for forbidden in forbidden_type_escapes:
        assert forbidden not in source


def test_router_uses_rclpy_logger_single_message_argument() -> None:
    source = read_package_file("fa_voice_command_router_py/router_node.py")

    assert "%s" not in source
    assert "Ignored command while inactive: {raw}" in source
    assert "Unhandled command (pass-through): {raw}" in source
