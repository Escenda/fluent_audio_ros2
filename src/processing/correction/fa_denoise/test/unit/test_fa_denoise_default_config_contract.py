from pathlib import Path

import yaml


def test_default_config_requires_explicit_dtln_models() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_denoise"]["ros__parameters"]

    assert params["enabled"] is True
    assert params["backend"] == "dtln_onnx"
    assert params["dtln"]["model_1_path"] == ""
    assert params["dtln"]["model_2_path"] == ""
