from pathlib import Path

import yaml

def package_root() -> Path:
    return Path(__file__).parents[2]

def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_deesser"]["ros__parameters"]

    assert params["input_topic"] == "fa_deesser/input"
    assert params["output_topic"] == "fa_deesser/output"
    assert params["input_stream_id"] == "audio/normalized/mic"
    assert params["output"]["stream_id"] == "audio/deessed/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["detector"]["cutoff_hz"] == 4500.0
    assert params["detector"]["threshold"] == 0.08
    assert params["detector"]["attenuation_db"] == -9.0
    assert 0.0 < params["detector"]["cutoff_hz"] < params["expected"]["sample_rate"] / 2
    assert 0.0 <= params["detector"]["threshold"] <= 1.0
    assert -120.0 <= params["detector"]["attenuation_db"] <= 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is True
    assert params["diagnostics"]["publish_period_ms"] == 1000
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is True
