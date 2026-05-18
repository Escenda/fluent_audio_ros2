from pathlib import Path
import sys

import numpy as np
import pytest

from fa_audio_embedding_py.backends.base import AudioEmbeddingRequest
from fa_audio_embedding_py.backends.factory import (
    AudioEmbeddingBackendSettings,
    build_audio_embedding_backend,
)


PACKAGE_ROOT = Path(__file__).parents[2]


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
    worker_path: Path,
    dimension: int = 4,
) -> AudioEmbeddingBackendSettings:
    return AudioEmbeddingBackendSettings(
        name="external_worker",
        command=sys.executable,
        model_id="embedding-test-model",
        model_path="",
        args=(
            str(worker_path),
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
        ),
        payload_encoding="float32le_raw",
        timeout_sec=1.0,
        workspace_dir=tmp_path / "workspace",
        cleanup_audio_files=True,
        dimension=dimension,
    )


def test_external_worker_pipeline_returns_finite_embedding(tmp_path: Path) -> None:
    worker_path = PACKAGE_ROOT / "test" / "fixtures" / "fake_audio_embedding_worker.py"
    backend = build_audio_embedding_backend(
        _settings(tmp_path, worker_path=worker_path, dimension=4)
    )

    result = backend.embed(_request())

    assert result.model_id == "embedding-test-model"
    assert result.embedding.dtype == np.float32
    assert result.embedding.shape == (4,)
    assert np.all(np.isfinite(result.embedding))
    assert list((tmp_path / "workspace").iterdir()) == []


def test_external_worker_pipeline_rejects_dimension_mismatch(tmp_path: Path) -> None:
    worker_path = tmp_path / "bad_dimension_worker.py"
    worker_path.write_text(
        "print('0.1 0.2')\n",
        encoding="utf-8",
    )
    backend = build_audio_embedding_backend(
        _settings(tmp_path, worker_path=worker_path, dimension=4)
    )

    with pytest.raises(RuntimeError, match="audio embedding dimension mismatch"):
        backend.embed(_request())


def test_external_worker_pipeline_rejects_non_finite_embedding(tmp_path: Path) -> None:
    worker_path = tmp_path / "non_finite_worker.py"
    worker_path.write_text(
        "print('0.1 nan 0.3 0.4')\n",
        encoding="utf-8",
    )
    backend = build_audio_embedding_backend(
        _settings(tmp_path, worker_path=worker_path, dimension=4)
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        backend.embed(_request())
