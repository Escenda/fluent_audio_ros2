from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def test_default_config_declares_explicit_plc_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_packet_loss_concealment_node"]["ros__parameters"]

    assert params["input_topic"] == "fa_packet_loss_concealment/input"
    assert params["output_topic"] == "fa_packet_loss_concealment/output"
    assert params["input_stream_id"] == "audio/stream/input"
    assert params["output"]["stream_id"] == "audio/stream/plc"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["plc"]["max_gap_frames"] == 3
    assert params["plc"]["attenuation_per_gap"] == 0.7
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
