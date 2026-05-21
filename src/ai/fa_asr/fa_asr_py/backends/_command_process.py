from __future__ import annotations

import json
import os
import shutil
import string
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fa_asr_py.backends.base import (
    ASR_AUDIO_ENCODING_FLOAT32LE,
    RESULT_FORMAT_PLAIN_TEXT,
    RESULT_FORMAT_SEGMENTS_JSON_V1,
    AsrBackendCapability,
    AsrRequest,
    AsrTranscript,
    AsrTranscriptSegment,
    build_asr_transcript,
    plain_text_to_asr_transcript,
    validate_result_format,
)


_PAYLOAD_ENCODING = "float32le_raw"
_ALLOWED_ARG_FIELDS = frozenset(("audio", "model", "language", "output", "sample_rate"))
_ALLOWED_HEALTH_ARG_FIELDS = frozenset(("model", "language"))
_ALLOWED_OUTPUT_PATH_FIELDS = frozenset(
    ("audio", "model", "language", "sample_rate", "session_id", "user_turn_id")
)
_REQUIRED_ARG_FIELDS = frozenset(("audio", "model", "sample_rate"))
_REQUIRED_HEALTH_ARG_FIELDS = frozenset(("model",))


@dataclass(frozen=True)
class CommandProcessEnvironmentVariable:
    name: str
    value: str


@dataclass(frozen=True)
class CommandProcessConfig:
    executable: str
    model: str
    language: str
    args: tuple[str, ...]
    health_args: tuple[str, ...]
    timeout_sec: float
    working_directory: Path | None
    output_text_path: str
    workspace_dir: Path
    cleanup_audio_files: bool
    payload_encoding: str
    result_format: str
    capability: AsrBackendCapability
    environment: tuple[CommandProcessEnvironmentVariable, ...] = ()


class _CommandProcessRunner:
    def __init__(self, config: CommandProcessConfig) -> None:
        self._config = config
        self._config.workspace_dir.mkdir(parents=True, exist_ok=True)
        if self._config.health_args:
            self._run_health_check()

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        self._validate_request(request)
        audio_path = self._config.workspace_dir / f"{time.time_ns()}_{request.user_turn_id}.f32"
        output_path = self._build_output_text_path(audio_path, request)
        try:
            self._write_audio_payload(audio_path, request)
            command = [self._config.executable]
            command.extend(self._format_transcription_args(audio_path, output_path, request))
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self._config.working_directory)
                    if self._config.working_directory is not None
                    else None,
                    env=self._build_subprocess_environment(),
                    capture_output=True,
                    text=True,
                    timeout=self._config.timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("ASR backend command timed out") from exc
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                raise RuntimeError(
                    f"ASR backend command failed: code={completed.returncode} stderr={stderr}"
                )

            backend_output = self._read_transcript(completed.stdout, output_path)
            return self._parse_backend_output(
                backend_output,
                sample_count=request.payload.sample_count,
            )
        finally:
            if self._config.cleanup_audio_files and audio_path.exists():
                audio_path.unlink()

    def _run_health_check(self) -> None:
        command = [self._config.executable]
        command.extend(self._format_health_args())
        try:
            completed = subprocess.run(
                command,
                cwd=str(self._config.working_directory)
                if self._config.working_directory is not None
                else None,
                env=self._build_subprocess_environment(),
                capture_output=True,
                text=True,
                timeout=self._config.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("ASR backend health check timed out") from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                f"ASR backend health check failed: code={completed.returncode} stderr={stderr}"
            )

    def _build_output_text_path(self, audio_path: Path, request: AsrRequest) -> Path | None:
        if not self._config.output_text_path:
            return None
        rendered = self._config.output_text_path.format(
            audio=str(audio_path),
            model=self._config.model,
            language=self._config.language,
            sample_rate=request.sample_rate,
            session_id=request.session_id,
            user_turn_id=request.user_turn_id,
        )
        path = Path(rendered).expanduser()
        if not path.is_absolute():
            path = self._config.workspace_dir / path
        return path

    def _format_transcription_args(
        self, audio_path: Path, output_path: Path | None, request: AsrRequest
    ) -> list[str]:
        output_value = "" if output_path is None else str(output_path)
        return [
            part.format(
                audio=str(audio_path),
                model=self._config.model,
                language=self._config.language,
                sample_rate=request.sample_rate,
                output=output_value,
            )
            for part in self._config.args
        ]

    def _format_health_args(self) -> list[str]:
        return [
            part.format(
                model=self._config.model,
                language=self._config.language,
            )
            for part in self._config.health_args
        ]

    @staticmethod
    def _read_transcript(stdout_text: str, output_path: Path | None) -> str:
        if output_path is not None:
            if not output_path.is_file():
                raise RuntimeError(f"ASR output text file was not created: {output_path}")
            return output_path.read_text(encoding="utf-8").strip()
        return stdout_text.strip()

    def _parse_backend_output(self, output_text: str, *, sample_count: int) -> AsrTranscript:
        if self._config.result_format == RESULT_FORMAT_PLAIN_TEXT:
            return plain_text_to_asr_transcript(output_text, sample_count=sample_count)
        if self._config.result_format == RESULT_FORMAT_SEGMENTS_JSON_V1:
            return self._parse_segments_json_v1(output_text, sample_count=sample_count)
        raise RuntimeError(f"unsupported backend.result_format: {self._config.result_format}")

    @staticmethod
    def _parse_segments_json_v1(output_text: str, *, sample_count: int) -> AsrTranscript:
        try:
            document = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ASR backend returned malformed segments_json_v1 JSON") from exc
        if type(document) is not dict:
            raise RuntimeError("ASR backend segments_json_v1 output must be a JSON mapping")
        if set(document.keys()) != {"result_format", "segments"}:
            raise RuntimeError(
                "ASR backend segments_json_v1 output must contain only result_format and segments"
            )
        if document["result_format"] != RESULT_FORMAT_SEGMENTS_JSON_V1:
            raise RuntimeError(
                "ASR backend segments_json_v1 result_format must be segments_json_v1"
            )

        segments_value = document["segments"]
        if type(segments_value) is not list:
            raise RuntimeError("ASR backend segments_json_v1 segments must be a list")
        if not segments_value:
            raise RuntimeError("ASR backend segments_json_v1 segments must not be empty")

        segments: list[AsrTranscriptSegment] = []
        for index, segment_value in enumerate(segments_value):
            if type(segment_value) is not dict:
                raise RuntimeError(
                    f"ASR backend segments_json_v1 segment {index} must be a mapping"
                )
            segment_keys = set(segment_value.keys())
            required_keys = {"start_sample", "end_sample", "text"}
            allowed_keys = required_keys.union({"speaker_label"})
            if not required_keys.issubset(segment_keys):
                raise RuntimeError(
                    f"ASR backend segments_json_v1 segment {index} is missing required fields"
                )
            if not segment_keys.issubset(allowed_keys):
                raise RuntimeError(
                    f"ASR backend segments_json_v1 segment {index} contains unsupported fields"
                )
            speaker_label = (
                segment_value["speaker_label"] if "speaker_label" in segment_value else None
            )
            segments.append(
                AsrTranscriptSegment(
                    start_sample=segment_value["start_sample"],
                    end_sample=segment_value["end_sample"],
                    text=segment_value["text"],
                    speaker_label=speaker_label,
                )
            )

        return build_asr_transcript(tuple(segments), sample_count=sample_count)

    @staticmethod
    def _validate_request(request: AsrRequest) -> None:
        request.payload.validate_matches(
            AsrBackendCapability(
                audio_encoding=ASR_AUDIO_ENCODING_FLOAT32LE,
                sample_rate_hz=request.payload.sample_rate_hz,
                channels=1,
                streaming=False,
                final_results_only=True,
            )
        )
        request.payload.float32_samples()

    def _write_audio_payload(self, path: Path, request: AsrRequest) -> None:
        if self._config.payload_encoding != _PAYLOAD_ENCODING:
            raise RuntimeError(f"unsupported ASR payload_encoding: {self._config.payload_encoding}")
        self._write_float32le_raw(path, request.payload.float32_samples())

    @staticmethod
    def _write_float32le_raw(path: Path, samples: np.ndarray) -> None:
        path.write_bytes(samples.astype("<f4", copy=False).tobytes())

    def _build_subprocess_environment(self) -> dict[str, str] | None:
        if not self._config.environment:
            return None
        subprocess_environment = os.environ.copy()
        for variable in self._config.environment:
            subprocess_environment[variable.name] = variable.value
        return subprocess_environment


def _load_model_path_command_config(
    *,
    command: str,
    model_path_value: str,
    language: str,
    args: tuple[str, ...],
    timeout_sec: float,
    working_directory_value: str,
    output_text_path: str,
    workspace_dir: Path,
    cleanup_audio_files: bool,
    result_format: str,
    health_args: tuple[str, ...] = (),
) -> CommandProcessConfig:
    if not model_path_value:
        raise RuntimeError("backend.model_path is required")
    try:
        model_path = Path(model_path_value).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"backend.model_path does not exist: {model_path_value}") from exc
    if not model_path.is_file():
        raise RuntimeError(f"backend.model_path does not exist: {model_path}")
    if not os.access(model_path, os.R_OK):
        raise RuntimeError(f"backend.model_path is not readable: {model_path}")
    return _load_command_config(
        command=command,
        model_value=str(model_path),
        language=language,
        args=args,
        health_args=health_args,
        require_health_args=False,
        timeout_sec=timeout_sec,
        working_directory_value=working_directory_value,
        output_text_path=output_text_path,
        workspace_dir=workspace_dir,
        cleanup_audio_files=cleanup_audio_files,
        result_format=result_format,
    )


def _load_model_id_command_config(
    *,
    command: str,
    model: str,
    language: str,
    args: tuple[str, ...],
    health_args: tuple[str, ...],
    timeout_sec: float,
    working_directory_value: str,
    output_text_path: str,
    workspace_dir: Path,
    cleanup_audio_files: bool,
    result_format: str,
) -> CommandProcessConfig:
    model_value = model.strip()
    if not model_value:
        raise RuntimeError("backend.model is required")
    return _load_command_config(
        command=command,
        model_value=model_value,
        language=language,
        args=args,
        health_args=health_args,
        require_health_args=True,
        timeout_sec=timeout_sec,
        working_directory_value=working_directory_value,
        output_text_path=output_text_path,
        workspace_dir=workspace_dir,
        cleanup_audio_files=cleanup_audio_files,
        result_format=result_format,
    )


def _load_command_config(
    *,
    command: str,
    model_value: str,
    language: str,
    args: tuple[str, ...],
    health_args: tuple[str, ...],
    require_health_args: bool,
    timeout_sec: float,
    working_directory_value: str,
    output_text_path: str,
    workspace_dir: Path,
    cleanup_audio_files: bool,
    result_format: str,
) -> CommandProcessConfig:
    executable = _resolve_executable(command)
    validated_result_format = validate_result_format(result_format)

    arg_fields = _validate_backend_args(args)
    _validate_backend_health_args(health_args, required=require_health_args)
    _validate_output_text_path(output_text_path)
    if "output" in arg_fields and not output_text_path:
        raise RuntimeError(
            "backend.output_text_path is required when backend.args uses the {output} placeholder"
        )

    if timeout_sec <= 0.0:
        raise RuntimeError("backend.timeout_sec must be greater than zero")

    try:
        working_directory = (
            Path(working_directory_value).expanduser().resolve(strict=True)
            if working_directory_value
            else None
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"backend.working_directory does not exist: {working_directory_value}"
        ) from exc
    if working_directory is not None and not working_directory.is_dir():
        raise RuntimeError(f"backend.working_directory does not exist: {working_directory}")

    return CommandProcessConfig(
        executable=executable,
        model=model_value,
        language=language,
        args=args,
        health_args=health_args,
        timeout_sec=timeout_sec,
        working_directory=working_directory,
        output_text_path=output_text_path,
        workspace_dir=workspace_dir,
        cleanup_audio_files=cleanup_audio_files,
        payload_encoding=_PAYLOAD_ENCODING,
        result_format=validated_result_format,
        capability=AsrBackendCapability(
            audio_encoding=ASR_AUDIO_ENCODING_FLOAT32LE,
            sample_rate_hz=0,
            channels=1,
            streaming=False,
            final_results_only=True,
        ),
    )


def _resolve_executable(command: str) -> str:
    if not command:
        raise RuntimeError("backend.command is required")
    command_path = Path(command).expanduser()
    if "/" in command or command_path.is_absolute():
        try:
            resolved_path = command_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"backend.command does not exist: {command_path}") from exc
        if not resolved_path.is_file():
            raise RuntimeError(f"backend.command is not a file: {resolved_path}")
        if not os.access(resolved_path, os.X_OK):
            raise RuntimeError(f"backend.command is not executable: {resolved_path}")
        return str(resolved_path)
    resolved = shutil.which(command)
    if not resolved:
        raise RuntimeError(f"backend.command not found in PATH: {command}")
    return str(Path(resolved).resolve(strict=True))


def _validate_backend_args(args: tuple[str, ...]) -> set[str]:
    if not args:
        raise RuntimeError("backend.args must not be empty")
    fields: set[str] = set()
    for part in args:
        fields.update(
            _parse_format_fields(
                value=part,
                allowed_fields=_ALLOWED_ARG_FIELDS,
                field_label="backend.args",
            )
        )
    missing = sorted(_REQUIRED_ARG_FIELDS.difference(fields))
    if missing:
        raise RuntimeError(
            "backend.args must include placeholders: "
            + ", ".join(f"{{{field}}}" for field in missing)
        )
    return fields


def _validate_backend_health_args(args: tuple[str, ...], *, required: bool) -> None:
    if not args:
        if required:
            raise RuntimeError("backend.health_args must not be empty")
        return
    fields: set[str] = set()
    for part in args:
        fields.update(
            _parse_format_fields(
                value=part,
                allowed_fields=_ALLOWED_HEALTH_ARG_FIELDS,
                field_label="backend.health_args",
            )
        )
    missing = sorted(_REQUIRED_HEALTH_ARG_FIELDS.difference(fields))
    if missing:
        raise RuntimeError(
            "backend.health_args must include placeholders: "
            + ", ".join(f"{{{field}}}" for field in missing)
        )


def _validate_output_text_path(output_text_path: str) -> None:
    if not output_text_path:
        return
    _parse_format_fields(
        value=output_text_path,
        allowed_fields=_ALLOWED_OUTPUT_PATH_FIELDS,
        field_label="backend.output_text_path",
    )


def _parse_format_fields(
    *,
    value: str,
    allowed_fields: frozenset[str],
    field_label: str,
) -> set[str]:
    formatter = string.Formatter()
    try:
        parsed_parts = tuple(formatter.parse(value))
    except ValueError as exc:
        raise RuntimeError(f"{field_label} contains malformed format string: {value}") from exc
    fields: set[str] = set()
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
    return fields
