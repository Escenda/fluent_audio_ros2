from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from fa_vad_py.backends.silero import SileroVAD
from fa_vad_py.contracts import (
    audio_frame_to_float_samples,
    validate_node_config,
    validate_qos_depth,
)


DEFAULT_ARGS = (
    "--audio",
    "{audio}",
    "--model",
    "{model}",
    "--provider",
    "{provider}",
    "--sample-rate",
    "{sample_rate}",
)


def _silero_backend(
    *,
    model_path: str,
    execution_provider: str = "cpu",
    command: str = "python3",
    args: tuple[str, ...] = DEFAULT_ARGS,
    timeout_sec: float = 1.0,
    workspace_dir: str = "/tmp/fluent_audio_fa_vad_test",
    sample_rate: int = 16000,
    frame_ms: int = 20,
    threshold_start: float = 0.5,
    threshold_end: float = 0.1,
    hangover_ms: int = 300,
) -> SileroVAD:
    return SileroVAD(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        hangover_ms=hangover_ms,
        model_path=model_path,
        execution_provider=execution_provider,
        command=command,
        args=args,
        timeout_sec=timeout_sec,
        workspace_dir=workspace_dir,
        cleanup_audio_files=True,
    )


def test_default_config_requires_explicit_silero_model_path() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default_vad.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_vad_node"]["ros__parameters"]

    assert params["backend.name"] == "silero"
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""
    assert params["backend.command"] == ""
    assert params["qos.depth"] == 10
    assert params["qos.reliable"] is False
    assert "{audio}" in params["backend.args"]
    assert "{model}" in params["backend.args"]
    assert "{provider}" in params["backend.args"]
    assert "{sample_rate}" in params["backend.args"]
    assert "silero" not in params


def test_vad_frame_contract_accepts_canonical_float32_mono() -> None:
    expected = np.array([-1.0, -0.25, 0.0, 0.5, 1.0], dtype=np.float32)

    samples = audio_frame_to_float_samples(
        data=expected.tobytes(),
        source_id="mic0",
        stream_id="stream0",
        layout="interleaved",
        channels=1,
        bit_depth=32,
    )

    np.testing.assert_allclose(samples, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"data": b""}, "AudioFrame data is required"),
        ({"source_id": ""}, "AudioFrame source_id and stream_id are required"),
        ({"stream_id": ""}, "AudioFrame source_id and stream_id are required"),
        ({"layout": "planar"}, "AudioFrame layout must be interleaved"),
        ({"channels": 2}, "AudioFrame channels must be 1"),
        ({"bit_depth": 16}, "AudioFrame bit_depth must be 32"),
        ({"data": b"1"}, "AudioFrame float32 data length is not byte-aligned"),
        (
            {"data": np.array([np.nan], dtype=np.float32).tobytes()},
            "AudioFrame contains non-finite samples",
        ),
        (
            {"data": np.array([1.25], dtype=np.float32).tobytes()},
            "AudioFrame samples must be normalized to \\[-1.0, 1.0\\]",
        ),
    ),
)
def test_vad_frame_contract_rejects_non_canonical_audio(
    kwargs: dict[str, bytes | str | int],
    message: str,
) -> None:
    base_kwargs = {
        "data": np.array([0.0], dtype=np.float32).tobytes(),
        "source_id": "mic0",
        "stream_id": "stream0",
        "layout": "interleaved",
        "channels": 1,
        "bit_depth": 32,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        audio_frame_to_float_samples(**base_kwargs)


def test_vad_node_source_does_not_hide_format_conversion() -> None:
    source_path = Path(__file__).parents[2] / "fa_vad_py" / "vad_node.py"
    source = source_path.read_text(encoding="utf-8")

    assert "_resample_linear" not in source
    assert "_convert_to_mono" not in source
    assert "np.clip" not in source
    assert "AudioFrame sample_rate must match target_sample_rate" in source


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"target_sample_rate": 0}, "target_sample_rate must be > 0"),
        ({"threshold_start": -0.1}, "threshold_start must be in \\[0.0, 1.0\\]"),
        ({"threshold_start": 1.1}, "threshold_start must be in \\[0.0, 1.0\\]"),
        ({"threshold_end": -0.1}, "threshold_end must be in \\[0.0, 1.0\\]"),
        ({"threshold_end": 1.1}, "threshold_end must be in \\[0.0, 1.0\\]"),
        ({"threshold_start": 0.2, "threshold_end": 0.8}, "threshold_end must be <= threshold_start"),
        ({"hangover_ms": 0}, "hangover_ms must be > 0"),
    ),
)
def test_vad_node_rejects_invalid_runtime_config_without_fallback(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    base_kwargs = {
        "target_sample_rate": 16000,
        "threshold_start": 0.5,
        "threshold_end": 0.1,
        "hangover_ms": 300,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(RuntimeError, match=message):
        validate_node_config(**base_kwargs)


def test_vad_node_rejects_invalid_qos_depth_without_fallback() -> None:
    assert validate_qos_depth(1) == 1
    with pytest.raises(RuntimeError, match="qos.depth must be > 0"):
        validate_qos_depth(0)
    with pytest.raises(RuntimeError, match="qos.depth must be > 0"):
        validate_qos_depth(-1)


def test_silero_backend_rejects_missing_model_path() -> None:
    with pytest.raises(RuntimeError, match="backend.model_path is required"):
        _silero_backend(model_path="")


def test_silero_backend_rejects_missing_model_path_directory() -> None:
    missing_repo = "/tmp/fluent_audio_missing_silero_repo"

    with pytest.raises(RuntimeError, match="backend.model_path does not exist"):
        _silero_backend(model_path=missing_repo)


def test_silero_backend_rejects_missing_execution_provider(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.execution_provider is required"):
        _silero_backend(model_path=str(tmp_path), execution_provider="")


def test_silero_backend_rejects_unknown_execution_provider(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported backend.execution_provider"):
        _silero_backend(model_path=str(tmp_path), execution_provider="tpu")


def test_silero_backend_rejects_missing_command(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.command is required"):
        _silero_backend(model_path=str(tmp_path), command="")


def test_silero_backend_rejects_missing_command_placeholders(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.args must include the \\{audio\\} placeholder"):
        _silero_backend(model_path=str(tmp_path), args=("--model", "{model}"))


def test_silero_backend_rejects_invalid_sample_rate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="sample_rate must be 8000 or 16000"):
        SileroVAD(
            sample_rate=44100,
            frame_ms=20,
            hangover_ms=300,
            threshold_start=0.5,
            threshold_end=0.1,
            model_path=str(tmp_path),
            execution_provider="cpu",
            command="python3",
            args=DEFAULT_ARGS,
            timeout_sec=1.0,
            workspace_dir="/tmp/fluent_audio_fa_vad_test",
            cleanup_audio_files=True,
        )


def test_silero_backend_rejects_invalid_thresholds(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="threshold_end must be <= threshold_start"):
        _silero_backend(
            model_path=str(tmp_path),
            threshold_start=0.2,
            threshold_end=0.8,
        )


def test_silero_backend_rejects_invalid_hangover(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="hangover_ms must be > 0"):
        _silero_backend(model_path=str(tmp_path), hangover_ms=0)


def test_silero_backend_rejects_implicit_hangover_rounding(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="hangover_ms must be >= frame_ms"):
        _silero_backend(model_path=str(tmp_path), hangover_ms=10, frame_ms=20)
    with pytest.raises(RuntimeError, match="hangover_ms must be divisible by frame_ms"):
        _silero_backend(model_path=str(tmp_path), hangover_ms=25, frame_ms=20)


def test_silero_backend_constructor_requires_explicit_vad_config() -> None:
    backend_path = Path(__file__).parents[2] / "fa_vad_py" / "backends" / "silero.py"
    source = backend_path.read_text(encoding="utf-8")

    assert "sample_rate: int = 16000" not in source
    assert "frame_ms: int = 20" not in source
    assert "hangover_ms: int = 250" not in source
    assert "threshold_start: float | None" not in source
    assert "threshold_end: float | None" not in source
    assert "VAD_THRESHOLD_START" not in source
    assert "VAD_THRESHOLD_END" not in source


def test_silero_backend_is_external_process_boundary() -> None:
    backend_path = Path(__file__).parents[2] / "fa_vad_py" / "backends" / "silero.py"
    node_path = Path(__file__).parents[2] / "fa_vad_py" / "vad_node.py"
    backend_source = backend_path.read_text(encoding="utf-8")
    node_source = node_path.read_text(encoding="utf-8")

    assert "import torch" not in backend_source
    assert "torch.hub" not in backend_source
    assert "subprocess.run" in backend_source
    assert 'declare_parameter("backend.command", "")' in node_source
    assert "VAD backend failed" in node_source


def test_silero_backend_runs_external_worker_contract(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    worker = Path(__file__).parents[1] / "fixtures" / "fake_vad_worker.py"
    backend = _silero_backend(
        model_path=str(model_dir),
        command=sys.executable,
        args=(
            str(worker),
            "--audio",
            "{audio}",
            "--model",
            "{model}",
            "--provider",
            "{provider}",
            "--sample-rate",
            "{sample_rate}",
        ),
        workspace_dir=str(workspace_dir),
    )

    result = backend.update(bytes(512 * 2))

    assert result.probability == 0.75
    assert result.is_speech is True
    assert result.start is True
    assert result.end is False
    assert list(workspace_dir.iterdir()) == []


def test_silero_worker_is_installed_by_cmake() -> None:
    cmake_path = Path(__file__).parents[2] / "CMakeLists.txt"
    worker_path = Path(__file__).parents[2] / "scripts" / "silero_vad_worker"

    assert worker_path.is_file()
    assert "scripts/silero_vad_worker" in cmake_path.read_text(encoding="utf-8")
