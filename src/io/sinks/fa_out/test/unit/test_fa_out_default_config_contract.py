from pathlib import Path

import yaml


def test_default_config_requires_explicit_sink_device() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_out"]["ros__parameters"]

    assert params["backend.name"] == "alsa_playback"
    assert params["audio.device_id"] == ""
