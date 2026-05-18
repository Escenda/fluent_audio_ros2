from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fa_audio_embedding_py.backends.base import AudioEmbeddingBackend
from fa_audio_embedding_py.backends.external_worker import (
    ExternalWorkerAudioEmbeddingBackend,
    ExternalWorkerAudioEmbeddingConfig,
)


@dataclass(frozen=True)
class AudioEmbeddingBackendSettings:
    name: str
    command: str
    model_id: str
    model_path: str
    args: tuple[str, ...]
    payload_encoding: str
    timeout_sec: float
    workspace_dir: Path
    cleanup_audio_files: bool
    dimension: int


def build_audio_embedding_backend(
    settings: AudioEmbeddingBackendSettings,
) -> AudioEmbeddingBackend:
    backend_name = settings.name.strip()
    if not backend_name:
        raise RuntimeError("backend.name is required")
    if backend_name == ExternalWorkerAudioEmbeddingBackend.name:
        return ExternalWorkerAudioEmbeddingBackend(
            ExternalWorkerAudioEmbeddingConfig(
                command=settings.command,
                model_id=settings.model_id,
                model_path=settings.model_path,
                args=settings.args,
                payload_encoding=settings.payload_encoding,
                timeout_sec=settings.timeout_sec,
                workspace_dir=settings.workspace_dir,
                cleanup_audio_files=settings.cleanup_audio_files,
                dimension=settings.dimension,
            )
        )
    raise RuntimeError(f"unsupported audio embedding backend.name: {backend_name}")
