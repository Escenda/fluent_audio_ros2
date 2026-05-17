from pathlib import Path

import yaml


def test_default_config_requires_explicit_model_and_provider() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_turn_detector"]["ros__parameters"]

    assert params["backend.name"] == "smart_turn_onnx"
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""


def test_turn_detector_node_rejects_non_canonical_audio_frames() -> None:
    package_root = Path(__file__).parents[2]
    source = (
        package_root / "fa_turn_detector_py" / "turn_detector_node.py"
    ).read_text(encoding="utf-8")

    assert "_resample_linear" not in source
    assert "_to_mono" not in source
    assert "np.frombuffer(bytes(msg.data), dtype=np.int16)" not in source
    assert "AudioFrame channels must be 1" in source
    assert "AudioFrame source_id and stream_id are required" in source
    assert "AudioFrame layout must be interleaved" in source
    assert "AudioFrame bit_depth must be 32" in source
    assert "AudioFrame sample_rate must match backend sample_rate" in source
    assert "AudioFrame samples must be normalized to [-1.0, 1.0]" in source


def test_smart_turn_backend_rejects_out_of_range_audio() -> None:
    package_root = Path(__file__).parents[2]
    source = (
        package_root / "fa_turn_detector_py" / "backends" / "smart_turn_onnx.py"
    ).read_text(encoding="utf-8")

    assert "audio = audio / max_abs" not in source
    assert "audio samples must be normalized to [-1.0, 1.0]" in source
