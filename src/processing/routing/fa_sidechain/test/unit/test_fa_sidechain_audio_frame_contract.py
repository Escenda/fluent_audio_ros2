from pathlib import Path

import yaml

def package_root() -> Path:
    return Path(__file__).parents[2]

def test_example_config_separates_topics_from_stream_ids() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_sidechain"]["ros__parameters"]

    assert params["sidechain_topic"] == "fa_sidechain/input"
    assert params["control_topic"] == "fa_sidechain/control"
    assert params["sidechain_stream_id"] == "audio/sidechain/frame"
    assert params["control"]["stream_id"] == "audio/sidechain/control"
    assert params["sidechain_topic"] != params["sidechain_stream_id"]
    assert params["control_topic"] != params["control"]["stream_id"]
    assert params["sidechain_stream_id"] != params["control"]["stream_id"]
    assert params["detector"]["threshold_rms"] == 0.05
    assert params["detector"]["active_gain_db"] == -12.0
    assert params["detector"]["inactive_gain_db"] == 0.0
    assert params["control"]["sample_rate"] == 1000
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
