from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def test_default_config_declares_required_jitter_buffer_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_jitter_buffer_node"]["ros__parameters"]

    assert params["input_topic"] == "fa_jitter_buffer/input"
    assert params["output_topic"] == "fa_jitter_buffer/output"
    assert params["input_stream_id"] == "audio/network/mic"
    assert params["output"]["stream_id"] == "audio/jitter_buffered/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["jitter"]["target_depth_frames"] == 2
    assert params["jitter"]["max_depth_frames"] == 8
    assert params["jitter"]["reset_on_epoch_regression"] is False
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
