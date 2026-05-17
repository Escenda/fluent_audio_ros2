from pathlib import Path
import sys

import pytest
import yaml

from fa_vad_py.backends.silero import SileroVAD


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
) -> SileroVAD:
    return SileroVAD(
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
    assert "{audio}" in params["backend.args"]
    assert "{model}" in params["backend.args"]
    assert "{provider}" in params["backend.args"]
    assert "{sample_rate}" in params["backend.args"]
    assert "silero" not in params


def test_vad_node_rejects_non_canonical_audio_frames() -> None:
    source_path = Path(__file__).parents[2] / "fa_vad_py" / "vad_node.py"
    source = source_path.read_text(encoding="utf-8")

    assert "_resample_linear" not in source
    assert "_convert_to_mono" not in source
    assert "np.clip" not in source
    assert "AudioFrame channels must be 1" in source
    assert "AudioFrame source_id and stream_id are required" in source
    assert "AudioFrame layout must be interleaved" in source
    assert "AudioFrame bit_depth must be 32" in source
    assert "AudioFrame sample_rate must match target_sample_rate" in source
    assert "AudioFrame samples must be normalized to [-1.0, 1.0]" in source


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
