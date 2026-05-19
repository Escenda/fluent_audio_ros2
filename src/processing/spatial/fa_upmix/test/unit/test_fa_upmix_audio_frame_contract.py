from pathlib import Path

import yaml

def test_default_config_requires_explicit_float32le_upmix_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_upmix"]["ros__parameters"]

    assert params["input_topic"] == "fa_upmix/input"
    assert params["output_topic"] == "fa_upmix/output"
    assert params["input_stream_id"] == "audio/mono/mic/raw"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["input_channels"] == 1
    assert params["output"]["stream_id"] == "audio/upmixed/mic/processed"
    assert params["output"]["channels"] == 2
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["mode"] == "mono_duplicate"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
