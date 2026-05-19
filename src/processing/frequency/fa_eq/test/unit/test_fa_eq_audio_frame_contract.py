from pathlib import Path

import yaml

def package_root() -> Path:
    return Path(__file__).parents[2]

def test_default_config_requires_float32_interleaved_three_band_eq_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_eq"]["ros__parameters"]

    assert params["input_topic"] == "fa_eq/input"
    assert params["output_topic"] == "fa_eq/output"
    assert params["input_stream_id"] == "audio/sample_format/mic"
    assert params["output"]["stream_id"] == "audio/eq/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["low"]["cutoff_hz"] == 250.0
    assert params["high"]["cutoff_hz"] == 4000.0
    assert 0.0 < params["low"]["cutoff_hz"] < params["high"]["cutoff_hz"]
    assert params["high"]["cutoff_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["gains"]["low_db"] == 0.0
    assert params["gains"]["mid_db"] == 0.0
    assert params["gains"]["high_db"] == 0.0
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
