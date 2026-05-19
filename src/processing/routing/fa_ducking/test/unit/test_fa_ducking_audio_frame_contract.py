from pathlib import Path

import yaml

def package_root() -> Path:
    return Path(__file__).parents[2]

def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_ducking"]["ros__parameters"]

    assert params["program_topic"] == "fa_ducking/program"
    assert params["sidechain_topic"] == "fa_ducking/sidechain"
    assert params["output_topic"] == "fa_ducking/output"
    assert params["program_stream_id"] == "audio/program/frame"
    assert params["sidechain_stream_id"] == "audio/sidechain/frame"
    assert params["output"]["stream_id"] == "audio/ducked/frame"
    assert params["program_topic"] != params["program_stream_id"]
    assert params["sidechain_topic"] != params["sidechain_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert len(
        {
            params["program_stream_id"],
            params["sidechain_stream_id"],
            params["output"]["stream_id"],
        }
    ) == 3
    assert params["sidechain"]["threshold_rms"] == 0.05
    assert params["sidechain"]["max_age_ms"] == 100
    assert params["ducking"]["gain_db"] == -12.0
    assert params["ducking"]["attack_ms"] == 10.0
    assert params["ducking"]["release_ms"] == 250.0
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
