from pathlib import Path

import yaml


def _package_root() -> Path:
    return Path(__file__).parents[2]


def test_default_config_declares_explicit_time_alignment_contract() -> None:
    config = yaml.safe_load((_package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_time_alignment"]["ros__parameters"]

    assert params["input_topic"] == "fa_time_alignment/input"
    assert params["output_topic"] == "fa_time_alignment/output"
    assert params["input_stream_id"] == "audio/frame_buffer/mic"
    assert params["output"]["stream_id"] == "audio/time_aligned/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["alignment"]["period_ms"] == 20.0
    assert params["alignment"]["phase_ms"] == 0.0
    assert params["alignment"]["max_adjust_ms"] == 2.0
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
