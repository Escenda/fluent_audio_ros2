from pathlib import Path

import yaml


def test_router_default_config_has_no_tts_playback_or_gain_controls() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_voice_command_router"]["ros__parameters"]

    assert "tts_play" not in params
    assert "tts_volume_db" not in params


def test_router_calls_tts_without_playback_or_gain_request() -> None:
    source_path = (
        Path(__file__).parents[2]
        / "fa_voice_command_router_py"
        / "router_node.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "request.play = False" in source
    assert "request.volume_db = 0.0" in source
    assert "tts_play" not in source
    assert "tts_volume_db" not in source
