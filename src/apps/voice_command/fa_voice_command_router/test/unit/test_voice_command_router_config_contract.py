from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_provides_all_required_startup_parameters() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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
