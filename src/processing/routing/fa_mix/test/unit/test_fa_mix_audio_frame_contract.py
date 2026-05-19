from pathlib import Path

import yaml

def package_root() -> Path:
    return Path(__file__).parents[2]

def test_example_config_separates_topics_from_stream_ids() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_mix"]["ros__parameters"]

    assert params["input_topics"] == ["fa_mix/tts"]
    assert params["input_stream_ids"] == ["audio/tts/frame"]
    assert params["output_topic"] == "fa_mix/output"
    assert params["output"]["stream_id"] == "audio/mix/output"
    assert params["input_topics"][0] != params["input_stream_ids"][0]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input_stream_ids"][0] != params["output"]["stream_id"]
    assert params["input_gains_db"] == [0.0]
    assert params["master_index"] == 0
    assert params["expected"]["sample_rate"] == 48000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["bit_depth"] == 16
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["layout"] == "interleaved"
    assert params["max_frame_age_ms"] == 60
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is True
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
