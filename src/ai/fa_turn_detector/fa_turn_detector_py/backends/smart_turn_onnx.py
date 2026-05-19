from __future__ import annotations

import math
import os
import shutil
import string
import subprocess
import time
from pathlib import Path

import numpy as np

from fa_turn_detector_py.backends.base import TurnDetectionResult


_PAYLOAD_ENCODING = "float32_npy"
_ALLOWED_ARG_FIELDS = frozenset(("audio", "model", "provider"))
_ALLOWED_HEALTH_ARG_FIELDS = frozenset(("model", "provider"))
_REQUIRED_ARG_FIELDS = frozenset(("audio", "model", "provider"))
_REQUIRED_HEALTH_ARG_FIELDS = frozenset(("model", "provider"))
_SUPPORTED_EXECUTION_PROVIDERS = frozenset((
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
))


class SmartTurnOnnxBackend:
    """External-process Smart Turn v3 ONNX backend adapter."""

    name = "smart_turn_onnx"
    sample_rate = 16000
    n_fft = 400
    hop_length = 160
    n_mels = 80
    n_frames = 800
    min_samples = sample_rate
    max_samples = hop_length * n_frames + n_fft

    def __init__(
        self,
        *,
        model_path: str,
        threshold: float,
        execution_provider: str,
        command: str,
        args: tuple[str, ...],
        health_args: tuple[str, ...],
        timeout_sec: float,
        workspace_dir: str,
        cleanup_audio_files: bool,
    ) -> None:
        self.model_path = self._validate_model_file(model_path)
        self._threshold = self._validate_threshold(threshold)
        self._execution_provider = self._validate_execution_provider(execution_provider)
        self._command = self._validate_command(command)
        self._args = self._validate_args(
            args=args,
            allowed_fields=_ALLOWED_ARG_FIELDS,
            required_fields=_REQUIRED_ARG_FIELDS,
            field_label="backend.args",
        )
        self._health_args = self._validate_args(
            args=health_args,
            allowed_fields=_ALLOWED_HEALTH_ARG_FIELDS,
            required_fields=_REQUIRED_HEALTH_ARG_FIELDS,
            field_label="backend.health_args",
        )
        self._timeout_sec = self._validate_timeout(timeout_sec)
        self._workspace_dir = self._validate_workspace_dir(workspace_dir)
        self._cleanup_audio_files = self._validate_cleanup_audio_files(
            cleanup_audio_files
        )
        self._payload_encoding = _PAYLOAD_ENCODING
        self._run_health_check()

    def detect(self, audio: np.ndarray) -> TurnDetectionResult:
        self._validate_audio(audio)
        audio_path = self._workspace_dir / f"{time.time_ns()}_turn_audio.npy"
        try:
            self._write_audio_payload(audio_path, audio)
            command = [self._command]
            command.extend(self._format_inference_args(self._args, audio_path=audio_path))
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("Smart Turn backend command timed out") from exc
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                raise RuntimeError(
                    "Smart Turn backend command failed: "
                    f"code={completed.returncode} stderr={stderr}"
                )
            probability = self._parse_probability(completed.stdout)
            return TurnDetectionResult(
                probability=probability,
                is_end=bool(probability >= self._threshold),
            )
        finally:
            if self._cleanup_audio_files and audio_path.exists():
                audio_path.unlink()

    def _run_health_check(self) -> None:
        command = [self._command]
        command.extend(self._format_health_args(self._health_args))
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Smart Turn backend health check timed out") from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                "Smart Turn backend health check failed: "
                f"code={completed.returncode} stderr={stderr}"
            )

    def _format_inference_args(
        self,
        args: tuple[str, ...],
        *,
        audio_path: Path,
    ) -> list[str]:
        return [
            item.format(
                audio=str(audio_path),
                model=str(self.model_path),
                provider=self._execution_provider,
            )
            for item in args
        ]

    def _format_health_args(self, args: tuple[str, ...]) -> list[str]:
        return [
            item.format(
                model=str(self.model_path),
                provider=self._execution_provider,
            )
            for item in args
        ]

    def _write_audio_payload(self, audio_path: Path, audio: np.ndarray) -> None:
        if self._payload_encoding != _PAYLOAD_ENCODING:
            raise RuntimeError(
                f"unsupported Smart Turn payload_encoding: {self._payload_encoding}"
            )
        np.save(audio_path, audio, allow_pickle=False)

    @staticmethod
    def _validate_audio(audio: np.ndarray) -> None:
        if audio.dtype != np.float32:
            raise ValueError("turn detector audio must be float32")
        if audio.ndim != 1:
            raise ValueError("turn detector audio must be one-dimensional")
        if audio.size == 0:
            raise ValueError("turn detector audio is required")
        if not np.all(np.isfinite(audio)):
            raise ValueError("turn detector audio contains non-finite samples")
        if np.any(audio < -1.0) or np.any(audio > 1.0):
            raise ValueError("turn detector audio samples must be normalized to [-1.0, 1.0]")

    @staticmethod
    def _validate_model_file(model_path: str) -> Path:
        path_value = model_path.strip()
        if not path_value:
            raise RuntimeError("backend.model_path is required")
        try:
            path = Path(path_value).expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Smart Turn model not found: {path_value}") from exc
        if not path.is_file():
            raise RuntimeError(f"Smart Turn model not found: {path}")
        if not os.access(path, os.R_OK):
            raise RuntimeError(f"Smart Turn model is not readable: {path}")
        if path.stat().st_size < 1024:
            header = path.read_text(encoding="utf-8", errors="ignore")[:128]
            if header.startswith("version https://git-lfs.github.com/spec"):
                raise RuntimeError(
                    f"Smart Turn model is a Git LFS pointer, not an ONNX file: {path}"
                )
            raise RuntimeError(f"Smart Turn model file is too small: {path}")
        return path

    @staticmethod
    def _validate_threshold(threshold: float) -> float:
        if not isinstance(threshold, float):
            raise RuntimeError("backend.threshold must be a double")
        if not math.isfinite(threshold):
            raise RuntimeError("backend.threshold must be finite")
        if threshold < 0.0 or threshold > 1.0:
            raise RuntimeError("backend.threshold must be between 0.0 and 1.0")
        return threshold

    @staticmethod
    def _validate_execution_provider(execution_provider: str) -> str:
        provider = execution_provider.strip()
        if not provider:
            raise RuntimeError("backend.execution_provider is required")
        if provider not in _SUPPORTED_EXECUTION_PROVIDERS:
            supported = ", ".join(sorted(_SUPPORTED_EXECUTION_PROVIDERS))
            raise RuntimeError(
                "backend.execution_provider must be one of: "
                f"{supported}; got {provider}"
            )
        return provider

    @staticmethod
    def _validate_command(command: str) -> str:
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
        if resolved is None:
            raise RuntimeError(f"backend.command not found on PATH: {command_value}")
        return str(Path(resolved).resolve(strict=True))

    @staticmethod
    def _validate_args(
        *,
        args: tuple[str, ...],
        allowed_fields: frozenset[str],
        required_fields: frozenset[str],
        field_label: str,
    ) -> tuple[str, ...]:
        if not isinstance(args, tuple):
            raise RuntimeError(f"{field_label} must be a tuple of non-empty strings")
        if not args:
            raise RuntimeError(f"{field_label} must not be empty")
        fields: set[str] = set()
        formatter = string.Formatter()
        for part in args:
            if not isinstance(part, str) or not part:
                raise RuntimeError(f"{field_label} must be a tuple of non-empty strings")
            try:
                parsed_parts = tuple(formatter.parse(part))
            except ValueError as exc:
                raise RuntimeError(f"{field_label} contains malformed format string: {part}") from exc
            for _, field_name, format_spec, conversion in parsed_parts:
                if field_name is None:
                    continue
                if conversion is not None or format_spec:
                    raise RuntimeError(
                        f"{field_label} placeholders must not use conversion or format spec"
                    )
                if field_name not in allowed_fields:
                    raise RuntimeError(f"unsupported {field_label} placeholder: {field_name}")
                fields.add(field_name)
        missing = sorted(required_fields.difference(fields))
        if missing:
            raise RuntimeError(
                f"{field_label} must include placeholders: "
                + ", ".join(f"{{{field}}}" for field in missing)
            )
        return args

    @staticmethod
    def _validate_timeout(timeout_sec: float) -> float:
        if not isinstance(timeout_sec, float):
            raise RuntimeError("backend.timeout_sec must be a double")
        if timeout_sec <= 0.0:
            raise RuntimeError("backend.timeout_sec must be greater than zero")
        return timeout_sec

    @staticmethod
    def _validate_cleanup_audio_files(cleanup_audio_files: bool) -> bool:
        if not isinstance(cleanup_audio_files, bool):
            raise RuntimeError("backend.cleanup_audio_files must be a bool")
        return cleanup_audio_files

    @staticmethod
    def _validate_workspace_dir(workspace_dir: str) -> Path:
        path_value = workspace_dir.strip()
        if not path_value:
            raise RuntimeError("backend.workspace_dir is required")
        path = Path(path_value).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _parse_probability(stdout_text: str) -> float:
        values = [line.strip() for line in stdout_text.splitlines() if line.strip()]
        if not values:
            raise RuntimeError("Smart Turn backend command returned empty stdout")
        try:
            probability = float(values[-1])
        except ValueError as exc:
            raise RuntimeError(
                "Smart Turn backend command must print probability as a float"
            ) from exc
        if not math.isfinite(probability):
            raise RuntimeError("Smart Turn backend probability must be finite")
        if probability < 0.0 or probability > 1.0:
            raise RuntimeError("Smart Turn backend probability must be in [0.0, 1.0]")
        return probability
