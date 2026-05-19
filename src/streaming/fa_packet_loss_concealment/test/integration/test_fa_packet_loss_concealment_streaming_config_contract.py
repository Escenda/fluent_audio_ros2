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
    assert isinstance(params["input_stream_id"], str)
    assert params["input_stream_id"]
    assert isinstance(params["output"]["stream_id"], str)
    assert params["output"]["stream_id"]
    assert params["input_stream_id"] != params["input_topic"]
    assert params["input_stream_id"] != params["output_topic"]
    assert params["output"]["stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert isinstance(params["qos"]["depth"], int)
    assert params["qos"]["depth"] > 0
    assert isinstance(params["qos"]["reliable"], bool)
    assert isinstance(params["diagnostics"]["publish_period_ms"], int)
    assert params["diagnostics"]["publish_period_ms"] > 0
