from pathlib import Path

import yaml


def test_default_config_requires_explicit_dtln_models() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_denoise"]["ros__parameters"]

    assert params["enabled"] is True
    assert params["backend.name"] == "dtln_onnx"
    assert "backend" not in params
    assert params["input_topic"] == "fa_denoise/input"
    assert params["output_topic"] == "fa_denoise/output"
    assert params["input_stream_id"] == "audio/resample16k/mic"
    assert params["output"]["stream_id"] == "audio/denoised/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert params["dtln"]["model_1_path"] == ""
    assert params["dtln"]["model_2_path"] == ""
