from pathlib import Path

import yaml

def package_root() -> Path:
    return Path(__file__).parents[2]

def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_declick"]["ros__parameters"]

    assert params["input_topic"] == "fa_declick/input"
    assert params["output_topic"] == "fa_declick/output"
    assert params["input_stream_id"] == "audio/noise_gated/mic"
    assert params["output"]["stream_id"] == "audio/declicked/mic"
    assert params["threshold"]["delta"] == 0.25
    assert 0.0 < params["threshold"]["delta"] <= 2.0
    assert params["window"]["max_samples"] == 1
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
