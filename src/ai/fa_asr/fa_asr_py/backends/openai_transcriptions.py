from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from fa_asr_py.backends._command_process import (
    CommandProcessConfig,
    CommandProcessEnvironmentVariable,
    _CommandProcessRunner,
    _load_model_id_command_config,
)
from fa_asr_py.backends.base import AsrRequest
from fa_asr_py.backends.openai_credentials import (
    OpenAiCredentialConfig,
    load_openai_credential_config,
)


@dataclass(frozen=True)
class OpenAiTranscriptionsAsrConfig:
    process: CommandProcessConfig
    credentials: OpenAiCredentialConfig


class OpenAiTranscriptionsAsrBackend:
    name = "openai_transcriptions"

    def __init__(self, config: OpenAiTranscriptionsAsrConfig) -> None:
        self._runner = _CommandProcessRunner(config.process)

    def transcribe(self, request: AsrRequest) -> str:
        return self._runner.transcribe(request)


def load_openai_transcriptions_config(
    *,
    command: str,
    model: str,
    api_key_env: str,
    language: str,
    args: tuple[str, ...],
    timeout_sec: float,
    working_directory_value: str,
    output_text_path: str,
    workspace_dir: Path,
    cleanup_audio_files: bool,
) -> OpenAiTranscriptionsAsrConfig:
    process = _load_model_id_command_config(
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
    credentials = load_openai_credential_config(
        parameter_name="backend.openai_transcriptions.api_key_env",
        api_key_env=api_key_env,
    )
    return OpenAiTranscriptionsAsrConfig(
        process=replace(
            process,
            environment=(
                CommandProcessEnvironmentVariable(
                    name="OPENAI_API_KEY",
                    value=credentials.api_key_value,
                ),
            ),
        ),
        credentials=credentials,
    )
