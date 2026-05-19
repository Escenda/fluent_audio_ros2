from pathlib import Path

import yaml

def test_default_config_requires_explicit_float32le_channel_convert_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_channel_convert"]["ros__parameters"]

    assert params["input_topic"] == "fa_channel_convert/input"
    assert params["output_topic"] == "fa_channel_convert/output"
    assert params["input_stream_id"] == "audio/sample_format/mic"
    assert params["input"]["channels"] == 1
    assert params["output"]["channels"] == 2
    assert params["output"]["stream_id"] == "audio/channel_converted/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["conversion"]["mode"] == "mono_to_stereo_duplicate"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
