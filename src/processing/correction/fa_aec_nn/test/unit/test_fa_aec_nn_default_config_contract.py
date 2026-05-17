from pathlib import Path

import yaml


def test_default_config_requires_explicit_backend() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_aec_nn"]["ros__parameters"]

    assert params["backend"] == ""
