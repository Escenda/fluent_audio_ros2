from pathlib import Path

import yaml

def package_root() -> Path:
    return Path(__file__).parents[2]

def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load(
        (package_root() / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_echo"]["ros__parameters"]

    assert params["input_topic"] == "fa_echo/input"
    assert params["output_topic"] == "fa_echo/output"
    assert params["input_stream_id"] == "audio/buffered/mic"
    assert params["output"]["stream_id"] == "audio/echo/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["output_topic"] != params["input_stream_id"]
    assert params["input_topic"] != params["output"]["stream_id"]
    assert params["echo"]["delay_ms"] == 250.0
    assert params["echo"]["feedback_gain"] == 0.35
    assert params["echo"]["wet_gain"] == 0.4
    assert params["echo"]["dry_gain"] == 0.8
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is True
