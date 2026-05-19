from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_defines_explicit_input_topic() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_record"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"
    assert params["input.qos.depth"] == 10
    assert params["input.qos.reliable"] is True
