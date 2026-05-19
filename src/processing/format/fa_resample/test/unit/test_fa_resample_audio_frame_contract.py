from pathlib import Path

import yaml

def package_root() -> Path:
    return Path(__file__).parents[2]

def test_default_config_requires_float32le_output_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_resample"]["ros__parameters"]

    assert params["target_sample_rate"] == 16000
    assert params["input"]["encoding"] == "FLOAT32LE"
    assert params["input"]["bit_depth"] == 32
    assert params["input"]["layout"] == "interleaved"
    assert params["output"]["encoding"] == "FLOAT32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["mic"]["input_topic"] == "audio/frame"
    assert params["mic"]["output_topic"] == "audio/resample16k/mic"
    assert params["mic"]["input_stream_id"] == "audio/float32/mic"
    assert params["mic"]["output"]["stream_id"] == "audio/preprocessed/mono16k"
    assert params["mic"]["input_topic"] != params["mic"]["input_stream_id"]
    assert params["mic"]["output_topic"] != params["mic"]["output"]["stream_id"]
