from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fa_asr_py.backends._command_process import (
    CommandProcessConfig,
    _CommandProcessRunner,
    _load_model_id_command_config,
)
from fa_asr_py.backends.base import AsrRequest


@dataclass(frozen=True)
class OpenAiRealtimeAsrConfig:
    process: CommandProcessConfig


class OpenAiRealtimeAsrBackend:
    name = "openai_realtime"

    def __init__(self, config: OpenAiRealtimeAsrConfig) -> None:
        self._runner = _CommandProcessRunner(config.process)

    def transcribe(self, request: AsrRequest) -> str:
        return self._runner.transcribe(request)


def load_openai_realtime_config(
    *,
    command: str,
    model: str,
    language: str,
    args: tuple[str, ...],
    timeout_sec: float,
    working_directory_value: str,
    output_text_path: str,
    workspace_dir: Path,
    cleanup_audio_files: bool,
) -> OpenAiRealtimeAsrConfig:
    return OpenAiRealtimeAsrConfig(
        process=_load_model_id_command_config(
            command=command,
            model=model,
            language=language,
            args=args,
            timeout_sec=timeout_sec,
            working_directory_value=working_directory_value,
            output_text_path=output_text_path,
            workspace_dir=workspace_dir,
            cleanup_audio_files=cleanup_audio_files,
        )
    )
