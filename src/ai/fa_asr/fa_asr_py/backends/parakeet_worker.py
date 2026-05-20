from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fa_asr_py.backends._command_process import (
    CommandProcessConfig,
    _CommandProcessRunner,
    _load_model_id_command_config,
)
from fa_asr_py.backends.base import AsrRequest, AsrTranscript


@dataclass(frozen=True)
class ParakeetWorkerAsrConfig:
    process: CommandProcessConfig


class ParakeetWorkerAsrBackend:
    name = "parakeet_worker"

    def __init__(self, config: ParakeetWorkerAsrConfig) -> None:
        self._runner = _CommandProcessRunner(config.process)

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        return self._runner.transcribe(request)


def load_parakeet_worker_config(
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
) -> ParakeetWorkerAsrConfig:
    return ParakeetWorkerAsrConfig(
        process=_load_model_id_command_config(
            command=command,
            model=model,
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
