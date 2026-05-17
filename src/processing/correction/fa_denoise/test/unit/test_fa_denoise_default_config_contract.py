from pathlib import Path

import yaml


def test_default_config_requires_explicit_dtln_models() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_denoise"]["ros__parameters"]

    assert params["enabled"] is True
    assert params["backend"] == "dtln_onnx"
    assert params["dtln"]["model_1_path"] == ""
    assert params["dtln"]["model_2_path"] == ""


def test_denoise_outputs_audio_frame_identity_without_analysis_metadata() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_denoise_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "msg.source_id.empty() || msg.stream_id.empty()" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert "out_msg.stream_id = config_.output_topic;" in source
    assert "out_msg.layout = kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source
