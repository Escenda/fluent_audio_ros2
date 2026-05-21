from __future__ import annotations

import base64
import json
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, TypeAlias

from fa_asr_py.backends.base import (
    ASR_AUDIO_ENCODING_FLOAT32LE,
    AsrAudioPayload,
    AsrBackendCapability,
    AsrRequest,
    AsrStreamRequest,
    AsrStreamResult,
    AsrTranscript,
    AsrTranscriptSegment,
    asr_transcript_text,
    build_asr_transcript,
)

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)

_AUDIO_PAYLOAD_ENCODING = "base64_float32le"
_HEALTH_OK_KEYS = frozenset(
    (
        "type",
        "model_class",
        "cache_aware_streaming",
        "sample_rate_hz",
        "channels",
        "audio_encoding",
        "streaming",
        "final_results_only",
        "supports_partials",
        "language",
        "chunk_size_samples",
        "max_partial_interval_ms",
    )
)
_STARTED_KEYS = frozenset(("type", "session_id"))
_AUDIO_ACCEPTED_KEYS = frozenset(("type", "session_id", "sample_count"))
_DRAINED_KEYS = frozenset(("type", "session_id"))
_FINISHED_KEYS = frozenset(("type", "session_id"))
_CANCELLED_KEYS = frozenset(("type", "session_id"))
_RESULT_KEYS = frozenset(("type", "session_id", "text", "sample_count"))


@dataclass(frozen=True)
class NemoRnntStreamingConfig:
    command: str
    model_path: Path
    language: str
    timeout_sec: float
    working_directory: Path | None
    sample_rate_hz: int
    channels: int
    chunk_size_samples: int
    emit_partial: bool
    max_partial_interval_ms: int


@dataclass(frozen=True)
class NemoRnntHealthCapability:
    model_class: str
    cache_aware_streaming: bool
    capability: AsrBackendCapability
    supports_partials: bool
    language: str
    chunk_size_samples: int
    max_partial_interval_ms: int


class NemoRnntStreamingAsrBackend:
    name = "nemo_rnnt_streaming"

    def __init__(self, config: NemoRnntStreamingConfig) -> None:
        self._config = config
        self._process = _JsonLinesWorkerProcess(config)
        try:
            health = self._process.health()
            _validate_health_capability(config, health)
        except Exception:
            self._process.terminate()
            raise
        self.capability = AsrBackendCapability(
            audio_encoding=ASR_AUDIO_ENCODING_FLOAT32LE,
            sample_rate_hz=config.sample_rate_hz,
            channels=config.channels,
            streaming=True,
            final_results_only=not config.emit_partial,
        )

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        raise RuntimeError("nemo_rnnt_streaming only supports streaming ASR")

    def start_stream(self, request: AsrStreamRequest) -> "NemoRnntStreamingSession":
        return NemoRnntStreamingSession(self._config, self._process, request)


class NemoRnntStreamingSession:
    def __init__(
        self,
        config: NemoRnntStreamingConfig,
        process: "_JsonLinesWorkerProcess",
        request: AsrStreamRequest,
    ) -> None:
        self._config = config
        self._process = process
        self._session_id = request.session_id
        self._user_turn_id = request.user_turn_id
        self._pushed_sample_count = 0
        self._finished = False
        self._cancelled = False
        self._last_non_empty_partial_hypothesis_text: str | None = None
        process.start_stream(config, request)

    def push_audio(self, payload: AsrAudioPayload) -> tuple[AsrStreamResult, ...]:
        self._ensure_open()
        payload.validate_matches(
            AsrBackendCapability(
                audio_encoding=ASR_AUDIO_ENCODING_FLOAT32LE,
                sample_rate_hz=self._config.sample_rate_hz,
                channels=self._config.channels,
                streaming=True,
                final_results_only=not self._config.emit_partial,
            )
        )
        payload.float32_samples()
        self._pushed_sample_count += payload.sample_count
        results = self._process.push_audio(
            session_id=self._session_id,
            payload=payload,
        )
        return self._validate_results(results, final_required=False)

    def drain_results(self) -> tuple[AsrStreamResult, ...]:
        self._ensure_open()
        results = self._process.drain_results(session_id=self._session_id)
        return self._validate_results(results, final_required=False)

    def finish(self) -> tuple[AsrStreamResult, ...]:
        self._ensure_open()
        results = self._process.finish_stream(session_id=self._session_id)
        self._finished = True
        return self._validate_results(results, final_required=True)

    def cancel(self) -> None:
        if self._finished or self._cancelled:
            return
        self._process.cancel_stream(session_id=self._session_id)
        self._cancelled = True

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError("ASR stream is already finished")
        if self._cancelled:
            raise RuntimeError("ASR stream is already cancelled")

    def _validate_results(
        self, results: tuple[AsrStreamResult, ...], *, final_required: bool
    ) -> tuple[AsrStreamResult, ...]:
        saw_final = False
        validated_results: list[AsrStreamResult] = []
        for result in results:
            if result.sample_count > self._pushed_sample_count:
                raise RuntimeError(
                    "ASR stream result sample_count exceeds pushed audio sample count"
                )
            committed_result = self._apply_streaming_commit_semantics(result)
            if committed_result.is_final:
                saw_final = True
            self._remember_non_empty_partial_hypothesis(committed_result)
            validated_results.append(committed_result)
        if final_required and not saw_final:
            raise RuntimeError("ASR stream finish did not return a final result")
        return tuple(validated_results)

    def _apply_streaming_commit_semantics(self, result: AsrStreamResult) -> AsrStreamResult:
        result_text = asr_transcript_text(result.transcript)
        if not result.is_final or result_text:
            return result
        if self._last_non_empty_partial_hypothesis_text is None:
            return result
        return AsrStreamResult(
            transcript=build_asr_transcript(
                (
                    AsrTranscriptSegment(
                        start_sample=0,
                        end_sample=result.sample_count,
                        text=self._last_non_empty_partial_hypothesis_text,
                    ),
                ),
                sample_count=result.sample_count,
            ),
            is_final=True,
            sample_count=result.sample_count,
        )

    def _remember_non_empty_partial_hypothesis(self, result: AsrStreamResult) -> None:
        if result.is_final:
            return
        result_text = asr_transcript_text(result.transcript)
        if result_text:
            self._last_non_empty_partial_hypothesis_text = result_text


class _JsonLinesWorkerProcess:
    def __init__(self, config: NemoRnntStreamingConfig) -> None:
        self._config = config
        self._process = subprocess.Popen(
            [config.command],
            cwd=str(config.working_directory) if config.working_directory is not None else None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("ASR worker process did not expose JSONL pipes")
        self._stdin: TextIO = self._process.stdin
        self._stdout: TextIO = self._process.stdout
        self._stdout_lines: queue.Queue[str] = queue.Queue()
        self._stdout_reader = threading.Thread(target=self._read_stdout_lines, daemon=True)
        self._stdout_reader.start()

    def health(self) -> NemoRnntHealthCapability:
        self._write_message(
            {
                "type": "health",
                "model_path": str(self._config.model_path),
                "language": self._config.language,
                "sample_rate_hz": self._config.sample_rate_hz,
                "channels": self._config.channels,
                "audio_encoding": ASR_AUDIO_ENCODING_FLOAT32LE,
                "streaming": True,
                "emit_partial": self._config.emit_partial,
                "chunk_size_samples": self._config.chunk_size_samples,
                "max_partial_interval_ms": self._config.max_partial_interval_ms,
            }
        )
        return _parse_health_ok(self._read_message())

    def start_stream(self, config: NemoRnntStreamingConfig, request: AsrStreamRequest) -> None:
        self._write_message(
            {
                "type": "start",
                "session_id": request.session_id,
                "user_turn_id": request.user_turn_id,
                "model_path": str(config.model_path),
                "language": config.language,
                "sample_rate_hz": config.sample_rate_hz,
                "channels": config.channels,
                "audio_encoding": ASR_AUDIO_ENCODING_FLOAT32LE,
                "streaming": True,
                "emit_partial": config.emit_partial,
                "chunk_size_samples": config.chunk_size_samples,
                "max_partial_interval_ms": config.max_partial_interval_ms,
            }
        )
        message = self._read_message()
        mapping = _require_mapping(message, "ASR worker start response")
        _reject_unsupported_keys(mapping, _STARTED_KEYS, "ASR worker start response")
        _require_type(mapping, "stream_started", "ASR worker start response")
        response_session_id = _require_string_field(
            mapping, "session_id", "ASR worker start response"
        )
        if response_session_id != request.session_id:
            raise RuntimeError("ASR worker start response session_id does not match request")

    def push_audio(self, *, session_id: str, payload: AsrAudioPayload) -> tuple[AsrStreamResult, ...]:
        encoded = base64.b64encode(payload.data).decode("ascii")
        self._write_message(
            {
                "type": "audio",
                "session_id": session_id,
                "encoding": _AUDIO_PAYLOAD_ENCODING,
                "sample_count": payload.sample_count,
                "data": encoded,
            }
        )
        return self._read_result_batch(
            session_id=session_id,
            terminal_type="audio_accepted",
            terminal_keys=_AUDIO_ACCEPTED_KEYS,
            terminal_label="ASR worker audio response",
        )

    def drain_results(self, *, session_id: str) -> tuple[AsrStreamResult, ...]:
        self._write_message({"type": "drain", "session_id": session_id})
        return self._read_result_batch(
            session_id=session_id,
            terminal_type="drained",
            terminal_keys=_DRAINED_KEYS,
            terminal_label="ASR worker drain response",
        )

    def finish_stream(self, *, session_id: str) -> tuple[AsrStreamResult, ...]:
        self._write_message({"type": "finish", "session_id": session_id})
        return self._read_result_batch(
            session_id=session_id,
            terminal_type="finished",
            terminal_keys=_FINISHED_KEYS,
            terminal_label="ASR worker finish response",
        )

    def cancel_stream(self, *, session_id: str) -> None:
        self._write_message({"type": "cancel", "session_id": session_id})
        message = self._read_message()
        mapping = _require_mapping(message, "ASR worker cancel response")
        _reject_unsupported_keys(mapping, _CANCELLED_KEYS, "ASR worker cancel response")
        _require_type(mapping, "cancelled", "ASR worker cancel response")
        response_session_id = _require_string_field(
            mapping, "session_id", "ASR worker cancel response"
        )
        if response_session_id != session_id:
            raise RuntimeError("ASR worker cancel response session_id does not match request")

    def terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()

    def _read_result_batch(
        self,
        *,
        session_id: str,
        terminal_type: str,
        terminal_keys: frozenset[str],
        terminal_label: str,
    ) -> tuple[AsrStreamResult, ...]:
        results: list[AsrStreamResult] = []
        while True:
            message = self._read_message()
            mapping = _require_mapping(message, terminal_label)
            message_type = _require_string_field(mapping, "type", terminal_label)
            if message_type == terminal_type:
                _reject_unsupported_keys(mapping, terminal_keys, terminal_label)
                response_session_id = _require_string_field(mapping, "session_id", terminal_label)
                if response_session_id != session_id:
                    raise RuntimeError(f"{terminal_label} session_id does not match request")
                return tuple(results)
            if message_type == "partial" or message_type == "final":
                results.append(_parse_result(mapping))
                continue
            raise RuntimeError(f"{terminal_label} returned unsupported type: {message_type}")

    def _write_message(self, message: dict[str, JsonValue]) -> None:
        _write_json_line(self._stdin, message)

    def _read_message(self) -> JsonValue:
        try:
            line = self._stdout_lines.get(timeout=self._config.timeout_sec)
        except queue.Empty as exc:
            raise TimeoutError("ASR worker JSONL response timed out")
        if line == "":
            stderr_text = ""
            if self._process.stderr is not None:
                stderr_text = self._process.stderr.read().strip()
            raise RuntimeError(f"ASR worker closed stdout before JSONL response: {stderr_text}")
        return _parse_json_line(line)

    def _read_stdout_lines(self) -> None:
        for line in self._stdout:
            self._stdout_lines.put(line)
        self._stdout_lines.put("")


def load_nemo_rnnt_streaming_config(
    *,
    command: str,
    model_path_value: str,
    language: str,
    timeout_sec: float,
    working_directory_value: str,
    sample_rate_hz: int,
    channels: int,
    chunk_size_samples: int,
    chunk_ms: int,
    emit_partial: bool,
    max_partial_interval_ms: int,
) -> NemoRnntStreamingConfig:
    executable = _resolve_executable(command)
    model_path = _resolve_model_path(model_path_value)
    language_value = language.strip()
    if not language_value:
        raise RuntimeError("backend.language is required")
    if timeout_sec <= 0.0:
        raise RuntimeError("backend.timeout_sec must be greater than zero")
    working_directory = _resolve_working_directory(working_directory_value)
    _validate_positive_int(sample_rate_hz, "backend.sample_rate_hz")
    if channels != 1:
        raise RuntimeError("backend.channels must be 1 for nemo_rnnt_streaming")
    _validate_positive_int(chunk_size_samples, "backend.chunk_size_samples")
    if chunk_ms != 0:
        raise RuntimeError(
            "backend.chunk_ms is not supported yet; set backend.chunk_size_samples explicitly"
        )
    if type(emit_partial) is not bool:
        raise RuntimeError("backend.emit_partial must be a bool")
    _validate_positive_int(max_partial_interval_ms, "backend.max_partial_interval_ms")
    return NemoRnntStreamingConfig(
        command=executable,
        model_path=model_path,
        language=language_value,
        timeout_sec=timeout_sec,
        working_directory=working_directory,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        chunk_size_samples=chunk_size_samples,
        emit_partial=emit_partial,
        max_partial_interval_ms=max_partial_interval_ms,
    )


def _validate_health_capability(
    config: NemoRnntStreamingConfig, health: NemoRnntHealthCapability
) -> None:
    model_class = health.model_class.strip().lower()
    if "rnnt" not in model_class and "transducer" not in model_class:
        raise RuntimeError("ASR worker health check rejected non-RNNT model")
    if not health.cache_aware_streaming:
        raise RuntimeError("ASR worker health check requires cache-aware streaming")
    expected = AsrBackendCapability(
        audio_encoding=ASR_AUDIO_ENCODING_FLOAT32LE,
        sample_rate_hz=config.sample_rate_hz,
        channels=config.channels,
        streaming=True,
        final_results_only=not config.emit_partial,
    )
    if health.capability != expected:
        raise RuntimeError("ASR worker health capability does not match backend config")
    if config.emit_partial and not health.supports_partials:
        raise RuntimeError("ASR worker health check does not support partial results")
    if health.language != config.language:
        raise RuntimeError("ASR worker health language does not match backend.language")
    if health.chunk_size_samples != config.chunk_size_samples:
        raise RuntimeError("ASR worker health chunk size does not match backend config")
    if health.max_partial_interval_ms != config.max_partial_interval_ms:
        raise RuntimeError("ASR worker health partial interval does not match backend config")


def _parse_health_ok(message: JsonValue) -> NemoRnntHealthCapability:
    mapping = _require_mapping(message, "ASR worker health response")
    _reject_unsupported_keys(mapping, _HEALTH_OK_KEYS, "ASR worker health response")
    _require_type(mapping, "health_ok", "ASR worker health response")
    return NemoRnntHealthCapability(
        model_class=_require_string_field(mapping, "model_class", "ASR worker health response"),
        cache_aware_streaming=_require_bool_field(
            mapping, "cache_aware_streaming", "ASR worker health response"
        ),
        capability=AsrBackendCapability(
            audio_encoding=_require_string_field(
                mapping, "audio_encoding", "ASR worker health response"
            ),
            sample_rate_hz=_require_int_field(
                mapping, "sample_rate_hz", "ASR worker health response"
            ),
            channels=_require_int_field(mapping, "channels", "ASR worker health response"),
            streaming=_require_bool_field(mapping, "streaming", "ASR worker health response"),
            final_results_only=_require_bool_field(
                mapping, "final_results_only", "ASR worker health response"
            ),
        ),
        supports_partials=_require_bool_field(
            mapping, "supports_partials", "ASR worker health response"
        ),
        language=_require_string_field(mapping, "language", "ASR worker health response"),
        chunk_size_samples=_require_int_field(
            mapping, "chunk_size_samples", "ASR worker health response"
        ),
        max_partial_interval_ms=_require_int_field(
            mapping, "max_partial_interval_ms", "ASR worker health response"
        ),
    )


def _parse_result(mapping: dict[str, JsonValue]) -> AsrStreamResult:
    _reject_unsupported_keys(mapping, _RESULT_KEYS, "ASR worker result")
    result_type = _require_string_field(mapping, "type", "ASR worker result")
    text = _require_result_text_field(
        mapping,
        "text",
        "ASR worker result",
        allow_empty=result_type == "final",
    )
    sample_count = _require_int_field(mapping, "sample_count", "ASR worker result")
    transcript = build_asr_transcript(
        (
            AsrTranscriptSegment(
                start_sample=0,
                end_sample=sample_count,
                text=text,
            ),
        ),
        sample_count=sample_count,
        allow_empty_text=result_type == "final",
    )
    return AsrStreamResult(
        transcript=transcript,
        is_final=result_type == "final",
        sample_count=sample_count,
    )


def _parse_json_line(line: str) -> JsonValue:
    try:
        decoded = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ASR worker returned malformed JSONL") from exc
    return _validate_json_value(decoded)


def _validate_json_value(value) -> JsonValue:
    if value is None:
        return None
    if type(value) is str or type(value) is int or type(value) is float or type(value) is bool:
        return value
    if type(value) is list:
        return [_validate_json_value(item) for item in value]
    if type(value) is dict:
        validated: dict[str, JsonValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RuntimeError("ASR worker JSON object keys must be strings")
            validated[key] = _validate_json_value(item)
        return validated
    raise RuntimeError("ASR worker returned unsupported JSON value")


def _write_json_line(stream: TextIO, message: dict[str, JsonValue]) -> None:
    stream.write(json.dumps(message, separators=(",", ":")) + "\n")
    stream.flush()


def _require_mapping(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if type(value) is not dict:
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _reject_unsupported_keys(
    mapping: dict[str, JsonValue], allowed_keys: frozenset[str], label: str
) -> None:
    unsupported = set(mapping.keys()).difference(allowed_keys)
    if unsupported:
        raise RuntimeError(f"{label} contains unsupported fields: {','.join(sorted(unsupported))}")


def _require_type(mapping: dict[str, JsonValue], expected_type: str, label: str) -> None:
    value = _require_string_field(mapping, "type", label)
    if value != expected_type:
        raise RuntimeError(f"{label} type must be {expected_type}, got {value}")


def _require_string_field(mapping: dict[str, JsonValue], field: str, label: str) -> str:
    if field not in mapping:
        raise RuntimeError(f"{label} missing required field: {field}")
    value = mapping[field]
    if type(value) is not str:
        raise RuntimeError(f"{label} field {field} must be a string")
    if field != "text" and not value.strip():
        raise RuntimeError(f"{label} field {field} must not be empty")
    return value


def _require_result_text_field(
    mapping: dict[str, JsonValue],
    field: str,
    label: str,
    *,
    allow_empty: bool,
) -> str:
    value = _require_string_field(mapping, field, label)
    if not allow_empty and not value.strip():
        raise RuntimeError("ASR backend returned an empty transcript")
    return value


def _require_int_field(mapping: dict[str, JsonValue], field: str, label: str) -> int:
    if field not in mapping:
        raise RuntimeError(f"{label} missing required field: {field}")
    value = mapping[field]
    if type(value) is not int:
        raise RuntimeError(f"{label} field {field} must be an integer")
    if value <= 0:
        raise RuntimeError(f"{label} field {field} must be greater than zero")
    return value


def _require_bool_field(mapping: dict[str, JsonValue], field: str, label: str) -> bool:
    if field not in mapping:
        raise RuntimeError(f"{label} missing required field: {field}")
    value = mapping[field]
    if type(value) is not bool:
        raise RuntimeError(f"{label} field {field} must be a bool")
    return value


def _resolve_executable(command: str) -> str:
    if not command.strip():
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


def _resolve_model_path(model_path_value: str) -> Path:
    if not model_path_value.strip():
        raise RuntimeError("backend.model_path is required")
    try:
        model_path = Path(model_path_value).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"backend.model_path does not exist: {model_path_value}") from exc
    if not model_path.is_file():
        raise RuntimeError(f"backend.model_path does not exist: {model_path}")
    if model_path.suffix != ".nemo":
        raise RuntimeError("backend.model_path must point to a local .nemo file")
    if not os.access(model_path, os.R_OK):
        raise RuntimeError(f"backend.model_path is not readable: {model_path}")
    return model_path


def _resolve_working_directory(value: str) -> Path | None:
    if not value.strip():
        return None
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"backend.working_directory does not exist: {value}") from exc
    if not path.is_dir():
        raise RuntimeError(f"backend.working_directory does not exist: {path}")
    return path


def _validate_positive_int(value: int, label: str) -> None:
    if type(value) is not int:
        raise RuntimeError(f"{label} must be an integer")
    if value <= 0:
        raise RuntimeError(f"{label} must be greater than zero")
