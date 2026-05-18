from __future__ import annotations

import os
import shutil
import string
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fa_audio_embedding_py.backends.base import (
    AudioEmbeddingRequest,
    AudioEmbeddingResult,
)


_PAYLOAD_ENCODING = "float32le_raw"
_ALLOWED_ARG_FIELDS = frozenset(
    ("audio", "model_id", "model_path", "sample_rate", "dimension", "source_id", "stream_id")
)
_REQUIRED_ARG_FIELDS = frozenset(("audio", "model_id", "sample_rate", "dimension"))


@dataclass(frozen=True)
class ExternalWorkerAudioEmbeddingConfig:
    command: str
    model_id: str
    model_path: str
    args: tuple[str, ...]
    timeout_sec: float
    workspace_dir: Path
    cleanup_audio_files: bool
    dimension: int
    payload_encoding: str = _PAYLOAD_ENCODING


class ExternalWorkerAudioEmbeddingBackend:
    name = "external_worker"

    def __init__(self, config: ExternalWorkerAudioEmbeddingConfig) -> None:
        self._config = _validate_config(config)

    def embed(self, request: AudioEmbeddingRequest) -> AudioEmbeddingResult:
        if request.sample_rate <= 0:
            raise ValueError("AudioEmbeddingRequest sample_rate must be positive")
        audio_path = self._config.workspace_dir / f"{time.time_ns()}_embedding.f32"
        try:
            self._write_audio_payload(audio_path, request.samples)
            command = [self._config.command]
            command.extend(self._format_args(audio_path, request))
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self._config.timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("audio embedding backend command timed out") from exc
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                raise RuntimeError(
                    "audio embedding backend command failed: "
                    f"code={completed.returncode} stderr={stderr}"
                )
            embedding = _parse_embedding_stdout(completed.stdout, self._config.dimension)
            return AudioEmbeddingResult(
                model_id=self._config.model_id,
                embedding=embedding,
            )
        finally:
            if self._config.cleanup_audio_files and audio_path.exists():
                audio_path.unlink()

    def _format_args(self, audio_path: Path, request: AudioEmbeddingRequest) -> list[str]:
        return [
            item.format(
                audio=str(audio_path),
                model_id=self._config.model_id,
                model_path=self._config.model_path,
                sample_rate=str(request.sample_rate),
                dimension=str(self._config.dimension),
                source_id=request.source_id,
                stream_id=request.stream_id,
            )
            for item in self._config.args
        ]

    def _write_audio_payload(self, audio_path: Path, samples: np.ndarray) -> None:
        if self._config.payload_encoding != _PAYLOAD_ENCODING:
            raise RuntimeError(
                f"unsupported audio embedding payload_encoding: {self._config.payload_encoding}"
            )
        audio_path.write_bytes(samples.astype("<f4", copy=False).tobytes())


def _validate_config(
    config: ExternalWorkerAudioEmbeddingConfig,
) -> ExternalWorkerAudioEmbeddingConfig:
    command = _resolve_executable(config.command)
    model_id = config.model_id.strip()
    if not model_id:
        raise RuntimeError("backend.model_id is required")
    model_path = _validate_model_path(config.model_path)
    _validate_args(config.args)
    if config.timeout_sec <= 0.0:
        raise RuntimeError("backend.timeout_sec must be > 0")
    if int(config.dimension) <= 0:
        raise RuntimeError("embedding.dimension must be > 0")
    workspace_dir = Path(config.workspace_dir).expanduser()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if config.payload_encoding != _PAYLOAD_ENCODING:
        raise RuntimeError(f"unsupported backend.payload_encoding: {config.payload_encoding}")
    return ExternalWorkerAudioEmbeddingConfig(
        command=command,
        model_id=model_id,
        model_path=model_path,
        args=config.args,
        timeout_sec=float(config.timeout_sec),
        workspace_dir=workspace_dir,
        cleanup_audio_files=bool(config.cleanup_audio_files),
        dimension=int(config.dimension),
        payload_encoding=config.payload_encoding,
    )


def _validate_model_path(model_path: str) -> str:
    model_path_value = model_path.strip()
    if not model_path_value:
        return ""
    try:
        resolved_path = Path(model_path_value).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"backend.model_path does not exist: {model_path_value}") from exc
    if not os.access(resolved_path, os.R_OK):
        raise RuntimeError(f"backend.model_path is not readable: {resolved_path}")
    return str(resolved_path)


def _resolve_executable(command: str) -> str:
    command_value = command.strip()
    if not command_value:
        raise RuntimeError("backend.command is required")
    command_path = Path(command_value).expanduser()
    if "/" in command_value or command_path.is_absolute():
        try:
            resolved_path = command_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"backend.command does not exist: {command_path}") from exc
        if not resolved_path.is_file():
            raise RuntimeError(f"backend.command is not a file: {resolved_path}")
        if not os.access(resolved_path, os.X_OK):
            raise RuntimeError(f"backend.command is not executable: {resolved_path}")
        return str(resolved_path)
    resolved = shutil.which(command_value)
    if not resolved:
        raise RuntimeError(f"backend.command not found in PATH: {command_value}")
    return str(Path(resolved).resolve(strict=True))


def _validate_args(args: tuple[str, ...]) -> None:
    if not args:
        raise RuntimeError("backend.args must not be empty")
    formatter = string.Formatter()
    fields: set[str] = set()
    for part in args:
        try:
            parsed_parts = tuple(formatter.parse(part))
        except ValueError as exc:
            raise RuntimeError(f"backend.args contains malformed format string: {part}") from exc
        for _, field_name, format_spec, conversion in parsed_parts:
            if field_name is None:
                continue
            if conversion is not None or format_spec:
                raise RuntimeError("backend.args placeholders must not use conversion or format spec")
            if field_name not in _ALLOWED_ARG_FIELDS:
                raise RuntimeError(f"unsupported backend.args placeholder: {field_name}")
            fields.add(field_name)
    missing = sorted(_REQUIRED_ARG_FIELDS.difference(fields))
    if missing:
        raise RuntimeError(
            "backend.args must include placeholders: "
            + ", ".join(f"{{{field}}}" for field in missing)
        )


def _parse_embedding_stdout(stdout_text: str, dimension: int) -> np.ndarray:
    tokens = stdout_text.strip().split()
    if not tokens:
        raise RuntimeError("audio embedding backend returned an empty embedding")
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise RuntimeError("audio embedding backend returned non-float values") from exc
    embedding = np.asarray(values, dtype=np.float32)
    if embedding.ndim != 1:
        raise RuntimeError("audio embedding backend returned non-vector output")
    if embedding.size != int(dimension):
        raise RuntimeError(
            f"audio embedding dimension mismatch: expected {dimension}, got {embedding.size}"
        )
    if not np.all(np.isfinite(embedding)):
        raise RuntimeError("audio embedding backend returned non-finite values")
    return embedding
