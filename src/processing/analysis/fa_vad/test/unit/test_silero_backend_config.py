from pathlib import Path

import pytest
import yaml

from fa_vad_py.backends.silero import SileroVAD


def test_default_config_requires_explicit_silero_model_path() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default_vad.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_vad_node"]["ros__parameters"]

    assert params["backend.name"] == "silero"
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""
    assert "silero" not in params


def test_vad_node_rejects_non_canonical_audio_frames() -> None:
    source_path = Path(__file__).parents[2] / "fa_vad_py" / "vad_node.py"
    source = source_path.read_text(encoding="utf-8")

    assert "_resample_linear" not in source
    assert "_convert_to_mono" not in source
    assert "np.clip" not in source
    assert "AudioFrame channels must be 1" in source
    assert "AudioFrame bit_depth must be 32" in source
    assert "AudioFrame sample_rate must match target_sample_rate" in source
    assert "AudioFrame samples must be normalized to [-1.0, 1.0]" in source


def test_silero_backend_rejects_missing_model_path() -> None:
    with pytest.raises(RuntimeError, match="backend.model_path is required"):
        SileroVAD(model_path="", execution_provider="cpu")


def test_silero_backend_rejects_missing_model_path_directory() -> None:
    missing_repo = "/tmp/fluent_audio_missing_silero_repo"

    with pytest.raises(RuntimeError, match="backend.model_path does not exist"):
        SileroVAD(model_path=missing_repo, execution_provider="cpu")


def test_silero_backend_rejects_missing_execution_provider(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.execution_provider is required"):
        SileroVAD(model_path=str(tmp_path), execution_provider="")


def test_silero_backend_rejects_unknown_execution_provider(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported backend.execution_provider"):
        SileroVAD(model_path=str(tmp_path), execution_provider="tpu")
