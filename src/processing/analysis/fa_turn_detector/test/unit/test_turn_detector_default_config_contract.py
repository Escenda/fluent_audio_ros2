from pathlib import Path

import yaml


def test_default_config_requires_explicit_model_and_provider() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_turn_detector"]["ros__parameters"]

    assert params["backend.name"] == "smart_turn_onnx"
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""
