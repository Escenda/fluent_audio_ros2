from pathlib import Path

import yaml

def package_root() -> Path:
    return Path(__file__).parents[2]

def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_agc"]["ros__parameters"]

    assert params["input_topic"] == "fa_agc/input"
    assert params["output_topic"] == "fa_agc/output"
    assert params["input_stream_id"] == "audio/compressed/mic"
    assert params["output"]["stream_id"] == "audio/agc/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["agc"]["target_rms"] == 0.1
    assert params["agc"]["min_gain"] == 0.25
    assert params["agc"]["max_gain"] == 4.0
    assert params["agc"]["attack_ms"] == 10.0
    assert params["agc"]["release_ms"] == 250.0
    assert 0.0 < params["agc"]["target_rms"] <= 1.0
    assert 0.0 < params["agc"]["min_gain"] <= 1.0
    assert params["agc"]["max_gain"] >= 1.0
    assert params["agc"]["min_gain"] <= params["agc"]["max_gain"]
    assert params["agc"]["attack_ms"] > 0.0
    assert params["agc"]["release_ms"] > 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
