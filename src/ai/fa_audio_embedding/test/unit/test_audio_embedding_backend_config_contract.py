import ast
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from fa_audio_embedding_py.backends.base import AudioEmbeddingRequest
from fa_audio_embedding_py.backends.factory import (
    AudioEmbeddingBackendSettings,
    build_audio_embedding_backend,
)


PACKAGE_ROOT = Path(__file__).parents[2]
PYTHON_SOURCES = tuple(
    path
    for path in sorted((PACKAGE_ROOT / "fa_audio_embedding_py").rglob("*.py"))
    if "__pycache__" not in path.parts
)


def _worker_path() -> Path:
    return PACKAGE_ROOT / "test" / "fixtures" / "fake_audio_embedding_worker.py"


def _request() -> AudioEmbeddingRequest:
    return AudioEmbeddingRequest(
        samples=np.asarray([0.0, 0.25, -0.5, 0.75], dtype=np.float32),
        sample_rate=16000,
        source_id="mic0",
        stream_id="audio/frame",
    )


def _settings(
    tmp_path: Path,
    *,
    backend_name: str = "external_worker",
    command: str = sys.executable,
    model_id: str = "embedding-test-model",
    model_path: str = "",
    args: tuple[str, ...] | None = None,
    payload_encoding: str = "float32le_raw",
    timeout_sec: float = 1.0,
    dimension: int = 4,
) -> AudioEmbeddingBackendSettings:
    backend_args = args
    if backend_args is None:
        backend_args = (
            str(_worker_path()),
            "--audio",
            "{audio}",
            "--model-id",
            "{model_id}",
            "--sample-rate",
            "{sample_rate}",
            "--dimension",
            "{dimension}",
            "--source-id",
            "{source_id}",
            "--stream-id",
            "{stream_id}",
        )
    return AudioEmbeddingBackendSettings(
        name=backend_name,
        command=command,
        model_id=model_id,
        model_path=model_path,
        args=backend_args,
        payload_encoding=payload_encoding,
        timeout_sec=timeout_sec,
        workspace_dir=tmp_path / "workspace",
        cleanup_audio_files=True,
        dimension=dimension,
    )


def test_default_config_requires_explicit_backend_identity_and_dimension() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    source = (PACKAGE_ROOT / "fa_audio_embedding_py" / "audio_embedding_node.py").read_text(
        encoding="utf-8"
    )
    params = config["fa_audio_embedding"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["backend.command"] == ""
    assert params["backend.model_id"] == ""
    assert params["backend.model_path"] == ""
    assert params["backend.args"] == []
    assert params["backend.payload_encoding"] == "float32le_raw"
    assert params["embedding.dimension"] == 0
    assert params["expected_source_id"] == ""
    assert params["expected_stream_id"] == ""
    assert 'declare_parameter("expected_source_id", Parameter.Type.STRING)' in source
    assert 'declare_parameter("expected_stream_id", Parameter.Type.STRING)' in source
    assert 'declare_parameter("backend.args", Parameter.Type.STRING_ARRAY)' in source
    assert 'declare_parameter("backend.payload_encoding", Parameter.Type.STRING)' in source
    assert "ParameterUninitializedException" in source


def test_build_backend_rejects_missing_backend_name(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.name is required"):
        build_audio_embedding_backend(_settings(tmp_path, backend_name=""))


def test_build_backend_rejects_unknown_backend_name(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported audio embedding backend.name"):
        build_audio_embedding_backend(_settings(tmp_path, backend_name="unknown"))


def test_external_worker_requires_model_id(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.model_id is required"):
        build_audio_embedding_backend(_settings(tmp_path, model_id=""))


def test_external_worker_requires_command(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.command is required"):
        build_audio_embedding_backend(_settings(tmp_path, command=""))


def test_external_worker_rejects_missing_command(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.command not found in PATH"):
        build_audio_embedding_backend(_settings(tmp_path, command="fa-audio-embedding-missing-worker"))


def test_external_worker_rejects_non_executable_command(tmp_path: Path) -> None:
    worker_path = tmp_path / "worker.py"
    worker_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    worker_path.chmod(0o644)

    with pytest.raises(RuntimeError, match="backend.command is not executable"):
        build_audio_embedding_backend(_settings(tmp_path, command=str(worker_path)))


def test_external_worker_rejects_invalid_model_path(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.model_path does not exist"):
        build_audio_embedding_backend(
            _settings(tmp_path, model_path=str(tmp_path / "missing-model.bin"))
        )


def test_external_worker_requires_model_path_arg_when_model_path_is_set(tmp_path: Path) -> None:
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"model")

    with pytest.raises(RuntimeError, match=r"\{model_path\}"):
        build_audio_embedding_backend(_settings(tmp_path, model_path=str(model_path)))


def test_external_worker_rejects_empty_args(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.args must not be empty"):
        build_audio_embedding_backend(_settings(tmp_path, args=()))


def test_external_worker_rejects_malformed_args(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.args contains malformed format string"):
        build_audio_embedding_backend(
            _settings(
                tmp_path,
                args=(
                    str(_worker_path()),
                    "--audio",
                    "{audio",
                    "--model-id",
                    "{model_id}",
                    "--sample-rate",
                    "{sample_rate}",
                    "--dimension",
                    "{dimension}",
                ),
            )
        )


def test_external_worker_requires_required_arg_placeholders(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"\{dimension\}"):
        build_audio_embedding_backend(
            _settings(
                tmp_path,
                args=(
                    str(_worker_path()),
                    "--audio",
                    "{audio}",
                    "--model-id",
                    "{model_id}",
                    "--sample-rate",
                    "{sample_rate}",
                ),
            )
        )


def test_external_worker_rejects_unknown_arg_placeholder(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported backend.args placeholder"):
        build_audio_embedding_backend(
            _settings(
                tmp_path,
                args=(
                    str(_worker_path()),
                    "--audio",
                    "{audio}",
                    "--model-id",
                    "{model_id}",
                    "--sample-rate",
                    "{sample_rate}",
                    "--dimension",
                    "{dimension}",
                    "--tenant",
                    "{tenant}",
                ),
            )
        )


def test_external_worker_rejects_unsupported_payload_encoding(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported backend.payload_encoding"):
        build_audio_embedding_backend(
            _settings(tmp_path, payload_encoding="pcm16le_raw")
        )


def test_external_worker_rejects_invalid_timeout_and_dimension(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.timeout_sec must be > 0"):
        build_audio_embedding_backend(_settings(tmp_path, timeout_sec=0.0))
    with pytest.raises(RuntimeError, match="embedding.dimension must be > 0"):
        build_audio_embedding_backend(_settings(tmp_path, dimension=0))


def test_audio_embedding_request_rejects_non_canonical_payloads() -> None:
    with pytest.raises(ValueError, match="source_id is required"):
        AudioEmbeddingRequest(
            samples=np.zeros(4, dtype=np.float32),
            sample_rate=16000,
            source_id="",
            stream_id="audio/frame",
        )
    with pytest.raises(ValueError, match="stream_id is required"):
        AudioEmbeddingRequest(
            samples=np.zeros(4, dtype=np.float32),
            sample_rate=16000,
            source_id="mic0",
            stream_id="",
        )
    with pytest.raises(ValueError, match="samples must be float32"):
        AudioEmbeddingRequest(
            samples=np.zeros(4, dtype=np.float64),
            sample_rate=16000,
            source_id="mic0",
            stream_id="audio/frame",
        )
    with pytest.raises(ValueError, match=r"normalized to \[-1.0, 1.0\]"):
        AudioEmbeddingRequest(
            samples=np.asarray([1.5], dtype=np.float32),
            sample_rate=16000,
            source_id="mic0",
            stream_id="audio/frame",
        )


def test_audio_embedding_python_sources_keep_dependency_boundary_explicit() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PYTHON_SOURCES)

    forbidden_tokens = (
        "ImportError",
        "from typing import Any",
        "dict[str, Any]",
        "Dict[str, Any]",
        ": Any",
        "-> Any",
        ": object",
        "-> object",
        "list[object]",
        "dict[str, object]",
        "# type: ignore",
    )
    for token in forbidden_tokens:
        assert token not in combined


def test_audio_embedding_backends_stay_ros_free() -> None:
    backend_files = tuple((PACKAGE_ROOT / "fa_audio_embedding_py" / "backends").glob("*.py"))
    assert backend_files
    forbidden_ros_tokens = (
        "rclpy",
        "fa_interfaces",
        "AudioFrame",
        "AudioEmbeddingFrame",
    )

    for backend_file in backend_files:
        source = backend_file.read_text(encoding="utf-8")
        for token in forbidden_ros_tokens:
            assert token not in source


def test_audio_embedding_node_rejects_non_canonical_audio_frames() -> None:
    source = (PACKAGE_ROOT / "fa_audio_embedding_py" / "audio_embedding_node.py").read_text(
        encoding="utf-8"
    )

    assert "np.zeros" not in source
    assert "_resample" not in source
    assert "_to_mono" not in source
    assert 'np.frombuffer(bytes(msg.data), dtype="<f4")' in source
    assert 'np.dtype("<f4").itemsize' in source
    assert "AudioFrame data is required" in source
    assert "AudioFrame source_id and stream_id are required" in source
    assert "AudioFrame source_id must match expected_source_id" in source
    assert "AudioFrame stream_id must match expected_stream_id" in source
    assert "AudioFrame layout must be" in source
    assert "AudioFrame encoding must be" in source
    assert "AudioFrame bit_depth must be" in source
    assert "AudioFrame sample_rate must match expected.sample_rate" in source
    assert "AudioFrame samples must be normalized to [-1.0, 1.0]" in source
    assert "source_id=msg.source_id" in source
    assert "stream_id=msg.stream_id" in source
    assert "out.stream_id = msg.stream_id" in source
    assert "out.stream_id = self.output_topic" not in source


def test_audio_embedding_node_drops_only_frame_contract_value_errors() -> None:
    source = (PACKAGE_ROOT / "fa_audio_embedding_py" / "audio_embedding_node.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    node_class = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "FaAudioEmbeddingNode"
    )
    on_audio = next(
        item
        for item in node_class.body
        if isinstance(item, ast.FunctionDef) and item.name == "on_audio"
    )
    try_blocks = [item for item in on_audio.body if isinstance(item, ast.Try)]

    assert len(try_blocks) == 2
    assert "_frame_to_float" in ast.dump(try_blocks[0])
    assert "backend" not in ast.dump(try_blocks[0])
    assert "embed" in ast.dump(try_blocks[1])
    backend_try = ast.dump(try_blocks[1])
    assert "ValueError" in backend_try
    assert "RuntimeError" in backend_try
