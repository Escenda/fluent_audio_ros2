from pathlib import Path

import yaml


def test_default_config_drops_reference_failures() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_aec_linear"]["ros__parameters"]

    assert params["reference_failure_policy"] == "drop"


def test_aec_linear_outputs_audio_frame_identity_without_analysis_metadata() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_aec_linear_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "msg.source_id.empty() || msg.stream_id.empty()" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert "out_msg.source_id = msg->source_id;" in source
    assert "out_msg.stream_id = config_.output_topic;" in source
    assert "out_msg.layout = kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source
