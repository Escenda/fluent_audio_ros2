from pathlib import Path

import yaml

def test_default_config_requires_explicit_pcm16_to_pcm32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_bit_depth"]["ros__parameters"]

    assert params["input_topic"] == "fa_bit_depth/input"
    assert params["output_topic"] == "fa_bit_depth/output"
    assert params["input_stream_id"] == "audio/raw/mic"
    assert params["output"]["stream_id"] == "audio/bit_depth/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["input"]["encoding"] == "PCM16LE"
    assert params["input"]["bit_depth"] == 16
    assert params["output"]["encoding"] == "PCM32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["layout"] == "interleaved"
