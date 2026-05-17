from pathlib import Path

import yaml


def test_default_config_has_no_playback_or_gain_parameters() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_tts"]["ros__parameters"]

    assert params["output_topic"] == "audio/tts/frame"
    assert "playback_topic" not in params
    assert "use_playback_topic" not in params
    assert "stop_topic" not in params
    assert "default_volume_db" not in params


def test_tts_node_does_not_publish_to_playback_topic() -> None:
    source_path = Path(__file__).parents[2] / "fa_tts_py" / "tts_node.py"
    source = source_path.read_text(encoding="utf-8")

    assert "playback_topic" not in source
    assert "use_playback_topic" not in source
    assert "play_pub" not in source
    assert "create_subscription(Empty" not in source
    assert "request.play is not supported by fa_tts" in source
    assert "request.volume_db is not supported by fa_tts" in source
