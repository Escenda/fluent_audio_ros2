from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fa_asr_py.backends._command_process import (
    CommandProcessConfig,
    _CommandProcessRunner,
    _load_model_path_command_config,
)
from fa_asr_py.backends.base import AsrRequest, AsrTranscript


@dataclass(frozen=True)
class WhisperCppAsrConfig:
    process: CommandProcessConfig


class WhisperCppAsrBackend:
    name = "whisper.cpp"

    def __init__(self, config: WhisperCppAsrConfig) -> None:
        self._runner = _CommandProcessRunner(config.process)

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        return self._runner.transcribe(request)


def load_whisper_cpp_config(
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
) -> WhisperCppAsrConfig:
    return WhisperCppAsrConfig(
        process=_load_model_path_command_config(
            command=command,
            model_path_value=model_path_value,
            language=language,
            args=args,
            health_args=health_args,
            timeout_sec=timeout_sec,
            working_directory_value=working_directory_value,
            output_text_path=output_text_path,
            workspace_dir=workspace_dir,
            cleanup_audio_files=cleanup_audio_files,
            result_format=result_format,
        )
    )
