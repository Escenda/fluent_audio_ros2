from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def test_default_config_requires_float32_interleaved_overlap_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_chunk_overlap"]["ros__parameters"]

    assert params["input_topic"] == "fa_chunk_overlap/input"
    assert params["output_topic"] == "fa_chunk_overlap/output"
    assert params["input_stream_id"] == "audio/float32le/mic"
    assert params["output"]["stream_id"] == "audio/chunked_overlap/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["window"]["frame_samples"] == 512
    assert params["window"]["hop_samples"] == 256
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is True
    assert params["diagnostics"]["publish_period_ms"] == 1000
