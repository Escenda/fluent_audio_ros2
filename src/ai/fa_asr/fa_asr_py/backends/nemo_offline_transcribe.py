from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from fa_asr_py.backends._command_process import (
    CommandProcessConfig,
    _CommandProcessRunner,
    _load_model_path_command_config,
)
from fa_asr_py.backends.base import (
    ASR_AUDIO_ENCODING_FLOAT32LE,
    AsrBackendCapability,
    AsrRequest,
    AsrTranscript,
)


@dataclass(frozen=True)
class NemoOfflineTranscribeAsrConfig:
    process: CommandProcessConfig


class NemoOfflineTranscribeAsrBackend:
    name = "nemo_offline_transcribe"

    def __init__(self, config: NemoOfflineTranscribeAsrConfig) -> None:
        self._runner = _CommandProcessRunner(config.process)
        self.capability = config.process.capability

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        request.payload.validate_matches(self.capability)
        request.payload.float32_samples()
        return self._runner.transcribe(request)


def load_nemo_offline_transcribe_config(
    *,
    command: str,
    model_path_value: str,
    language: str,
    timeout_sec: float,
    working_directory_value: str,
    output_text_path: str,
    workspace_dir: Path,
    cleanup_audio_files: bool,
    result_format: str,
    sample_rate_hz: int,
    channels: int,
) -> NemoOfflineTranscribeAsrConfig:
    _validate_backend_language(language)
    _validate_local_nemo_model_path(model_path_value)
    _validate_audio_contract(sample_rate_hz=sample_rate_hz, channels=channels)

    transcription_args = _default_transcription_args(
        result_format=result_format,
        output_text_path=output_text_path,
    )
    health_args = (
        "health",
        "--model",
        "{model}",
        "--language",
        "{language}",
        "--sample-rate",
        str(sample_rate_hz),
        "--channels",
        str(channels),
    )
    process = _load_model_path_command_config(
        command=command,
        model_path_value=model_path_value,
        language=language,
        args=transcription_args,
        health_args=health_args,
        timeout_sec=timeout_sec,
        working_directory_value=working_directory_value,
        output_text_path=output_text_path,
        workspace_dir=workspace_dir,
        cleanup_audio_files=cleanup_audio_files,
        result_format=result_format,
    )
    return NemoOfflineTranscribeAsrConfig(
        process=replace(
            process,
            capability=AsrBackendCapability(
                audio_encoding=ASR_AUDIO_ENCODING_FLOAT32LE,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                streaming=False,
                final_results_only=True,
            ),
        )
    )


def _default_transcription_args(
    *,
    result_format: str,
    output_text_path: str,
) -> tuple[str, ...]:
    args = (
        "transcribe",
        "--audio",
        "{audio}",
        "--model",
        "{model}",
        "--language",
        "{language}",
        "--sample-rate",
        "{sample_rate}",
        "--channels",
        "1",
        "--result-format",
        result_format,
    )
    if not output_text_path:
        return args
    return args + ("--output", "{output}")


def _validate_backend_language(language: str) -> None:
    if not language.strip():
        raise RuntimeError("backend.language is required")


def _validate_local_nemo_model_path(model_path_value: str) -> None:
    if not model_path_value.strip():
        raise RuntimeError("backend.model_path is required")
    if Path(model_path_value).expanduser().suffix != ".nemo":
        raise RuntimeError("backend.model_path must point to a local .nemo file")


def _validate_audio_contract(*, sample_rate_hz: int, channels: int) -> None:
    if type(sample_rate_hz) is not int or sample_rate_hz <= 0:
        raise RuntimeError("backend.sample_rate_hz must be a positive integer")
    if type(channels) is not int:
        raise RuntimeError("backend.channels must be an integer")
    if channels != 1:
        raise RuntimeError("backend.channels must be 1 for nemo_offline_transcribe")
