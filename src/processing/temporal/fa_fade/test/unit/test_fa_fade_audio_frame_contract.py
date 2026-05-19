from pathlib import Path

import yaml

def package_root() -> Path:
    return Path(__file__).parents[2]

def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_fade"]["ros__parameters"]

    assert params["input_topic"] == "fa_fade/input"
    assert params["output_topic"] == "fa_fade/output"
    assert params["input_stream_id"] == "audio/buffered/mic"
    assert params["output"]["stream_id"] == "audio/faded/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["fade"]["mode"] == "fade_in"
    assert params["fade"]["duration_frames"] == 16000
    assert params["fade"]["initial_position_frames"] == 0
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
