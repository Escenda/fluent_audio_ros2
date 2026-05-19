from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def test_default_config_declares_required_clock_drift_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_clock_drift"]["ros__parameters"]

    assert params["input_topic"] == "fa_clock_drift/input"
    assert params["output_topic"] == "fa_clock_drift/output"
    assert params["input_stream_id"] == "audio/sample_format/mic"
    assert params["output"]["stream_id"] == "audio/clock_drift_corrected/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["drift"]["ema_alpha"] == 0.1
    assert params["drift"]["max_correction_ms_per_frame"] == 2.0
    assert params["drift"]["reset_threshold_ms"] == 50.0
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
