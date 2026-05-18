from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fa_asr_py.backends.base import AsrBackend
from fa_asr_py.backends.local_command import (
    LocalCommandAsrBackend,
    load_local_command_config,
)
from fa_asr_py.backends.openai_realtime import (
    OpenAiRealtimeAsrBackend,
    load_openai_realtime_config,
)
from fa_asr_py.backends.openai_transcriptions import (
    OpenAiTranscriptionsAsrBackend,
    load_openai_transcriptions_config,
)
from fa_asr_py.backends.parakeet_worker import (
    ParakeetWorkerAsrBackend,
    load_parakeet_worker_config,
)
from fa_asr_py.backends.whisper_cpp import WhisperCppAsrBackend, load_whisper_cpp_config


@dataclass(frozen=True)
class AsrBackendSettings:
    name: str
    command: str
    model: str
    model_path: str
    openai_realtime_api_key_env: str
    openai_transcriptions_api_key_env: str
    language: str
    args: tuple[str, ...]
    health_args: tuple[str, ...]
    timeout_sec: float
    working_directory: str
    output_text_path: str
    workspace_dir: Path
    cleanup_audio_files: bool


def build_asr_backend(settings: AsrBackendSettings) -> AsrBackend:
    backend_name = settings.name.strip()
    if not backend_name:
        raise RuntimeError("backend.name is required")
    if backend_name == LocalCommandAsrBackend.name:
        return LocalCommandAsrBackend(
            load_local_command_config(
                command=settings.command.strip(),
                model_path_value=settings.model_path.strip(),
                language=settings.language,
                args=settings.args,
                health_args=settings.health_args,
                timeout_sec=settings.timeout_sec,
                working_directory_value=settings.working_directory.strip(),
                output_text_path=settings.output_text_path.strip(),
                workspace_dir=settings.workspace_dir,
                cleanup_audio_files=settings.cleanup_audio_files,
            )
        )
    if backend_name == WhisperCppAsrBackend.name:
        return WhisperCppAsrBackend(
            load_whisper_cpp_config(
                command=settings.command.strip(),
                model_path_value=settings.model_path.strip(),
                language=settings.language,
                args=settings.args,
                health_args=settings.health_args,
                timeout_sec=settings.timeout_sec,
                working_directory_value=settings.working_directory.strip(),
                output_text_path=settings.output_text_path.strip(),
                workspace_dir=settings.workspace_dir,
                cleanup_audio_files=settings.cleanup_audio_files,
            )
        )
    if backend_name == ParakeetWorkerAsrBackend.name:
        return ParakeetWorkerAsrBackend(
            load_parakeet_worker_config(
                command=settings.command.strip(),
                model=settings.model.strip(),
                language=settings.language,
                args=settings.args,
                health_args=settings.health_args,
                timeout_sec=settings.timeout_sec,
                working_directory_value=settings.working_directory.strip(),
                output_text_path=settings.output_text_path.strip(),
                workspace_dir=settings.workspace_dir,
                cleanup_audio_files=settings.cleanup_audio_files,
            )
        )
    if backend_name == OpenAiRealtimeAsrBackend.name:
        return OpenAiRealtimeAsrBackend(
            load_openai_realtime_config(
                command=settings.command.strip(),
                model=settings.model.strip(),
                api_key_env=settings.openai_realtime_api_key_env.strip(),
                language=settings.language,
                args=settings.args,
                health_args=settings.health_args,
                timeout_sec=settings.timeout_sec,
                working_directory_value=settings.working_directory.strip(),
                output_text_path=settings.output_text_path.strip(),
                workspace_dir=settings.workspace_dir,
                cleanup_audio_files=settings.cleanup_audio_files,
            )
        )
    if backend_name == OpenAiTranscriptionsAsrBackend.name:
        return OpenAiTranscriptionsAsrBackend(
            load_openai_transcriptions_config(
                command=settings.command.strip(),
                model=settings.model.strip(),
                api_key_env=settings.openai_transcriptions_api_key_env.strip(),
                language=settings.language,
                args=settings.args,
                health_args=settings.health_args,
                timeout_sec=settings.timeout_sec,
                working_directory_value=settings.working_directory.strip(),
                output_text_path=settings.output_text_path.strip(),
                workspace_dir=settings.workspace_dir,
                cleanup_audio_files=settings.cleanup_audio_files,
            )
        )
    raise RuntimeError(f"unsupported ASR backend.name: {backend_name}")
