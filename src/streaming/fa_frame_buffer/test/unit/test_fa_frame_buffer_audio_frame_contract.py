from pathlib import Path

import yaml


def test_default_config_requires_float32_interleaved_fixed_chunk_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_frame_buffer"]["ros__parameters"]

    assert params["input_topic"] == "fa_frame_buffer/input"
    assert params["output_topic"] == "fa_frame_buffer/output"
    assert params["input_stream_id"] == "audio/noise_gated/mic"
    assert params["output"]["stream_id"] == "audio/buffered/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["buffering"]["frames_per_chunk"] == 512
    assert params["buffering"]["max_buffered_chunks"] == 4
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
