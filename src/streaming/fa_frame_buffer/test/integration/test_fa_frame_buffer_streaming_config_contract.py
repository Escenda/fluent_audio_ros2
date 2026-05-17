from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def test_default_config_defines_explicit_streaming_data_plane_contract() -> None:
    package_name = package_root().name
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    assert len(config) == 1
    node_config = next(iter(config.values()))
    params = node_config["ros__parameters"]

    assert isinstance(params["input_topic"], str)
    assert params["input_topic"]
    assert isinstance(params["output_topic"], str)
    assert params["output_topic"]
    assert params["input_topic"] != params["output_topic"]
    assert isinstance(params["qos"]["depth"], int)
    assert params["qos"]["depth"] > 0
    assert isinstance(params["qos"]["reliable"], bool)
    assert isinstance(params["diagnostics"]["publish_period_ms"], int)
    assert params["diagnostics"]["publish_period_ms"] > 0


def test_streaming_package_has_runtime_contract_sources() -> None:
    package_name = package_root().name
    source = (package_root() / "src" / f"{package_name}_node.cpp").read_text(encoding="utf-8")

    assert "AudioFrame" in source
    assert "validate" in source
    assert "diagnostic" in source.lower()
