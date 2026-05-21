from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fa_asr_py.backends.base import AsrBackend
from fa_asr_py.backends.local_command import (
    LocalCommandAsrBackend,
    load_local_command_config,
)
from fa_asr_py.backends.nemo_rnnt_streaming import (
    NemoRnntStreamingAsrBackend,
    load_nemo_rnnt_streaming_config,
)
from fa_asr_py.backends.nemo_offline_transcribe import (
    NemoOfflineTranscribeAsrBackend,
    load_nemo_offline_transcribe_config,
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
    workspace_dir: Path = Path(".")
    cleanup_audio_files: bool = False
    command: str = ""
    model: str = ""
    model_path: str = ""
    openai_realtime_api_key_env: str = ""
    openai_transcriptions_api_key_env: str = ""
    language: str = ""
    args: tuple[str, ...] = ()
    health_args: tuple[str, ...] = ()
    timeout_sec: float = 0.0
    working_directory: str = ""
    output_text_path: str = ""
    result_format: str = ""
    sample_rate_hz: int = 0
    channels: int = 0
    chunk_size_samples: int = 0
    chunk_ms: int = 0
    emit_partial: bool = True
    max_partial_interval_ms: int = 0


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
                result_format=settings.result_format.strip(),
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
                result_format=settings.result_format.strip(),
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
                result_format=settings.result_format.strip(),
            )
        )
    if backend_name == NemoRnntStreamingAsrBackend.name:
        return NemoRnntStreamingAsrBackend(
            load_nemo_rnnt_streaming_config(
                command=settings.command.strip(),
                model_path_value=settings.model_path.strip(),
                language=settings.language,
                timeout_sec=settings.timeout_sec,
                working_directory_value=settings.working_directory.strip(),
                sample_rate_hz=settings.sample_rate_hz,
                channels=settings.channels,
                chunk_size_samples=settings.chunk_size_samples,
                chunk_ms=settings.chunk_ms,
                emit_partial=settings.emit_partial,
                max_partial_interval_ms=settings.max_partial_interval_ms,
            )
        )
    if backend_name == NemoOfflineTranscribeAsrBackend.name:
        return NemoOfflineTranscribeAsrBackend(
            load_nemo_offline_transcribe_config(
                command=settings.command.strip(),
                model_path_value=settings.model_path.strip(),
                language=settings.language,
                timeout_sec=settings.timeout_sec,
                working_directory_value=settings.working_directory.strip(),
                output_text_path=settings.output_text_path.strip(),
                workspace_dir=settings.workspace_dir,
                cleanup_audio_files=settings.cleanup_audio_files,
                result_format=settings.result_format.strip(),
                sample_rate_hz=settings.sample_rate_hz,
                channels=settings.channels,
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
                result_format=settings.result_format.strip(),
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
                result_format=settings.result_format.strip(),
            )
        )
    raise RuntimeError(f"unsupported ASR backend.name: {backend_name}")
