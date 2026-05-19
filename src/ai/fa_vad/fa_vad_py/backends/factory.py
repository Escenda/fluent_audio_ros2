from __future__ import annotations

from dataclasses import dataclass

from fa_vad_py.backends.base import VADBackend
from fa_vad_py.backends.silero import SileroVAD


@dataclass(frozen=True)
class VadBackendSettings:
    name: str
    sample_rate: int
    frame_ms: int
    window_samples: int
    history_buffer_ms: int
    hangover_ms: int
    threshold_start: float
    threshold_end: float
    model_path: str
    execution_provider: str
    command: str
    args: tuple[str, ...]
    timeout_sec: float
    workspace_dir: str
    cleanup_audio_files: bool


def build_vad_backend(settings: VadBackendSettings) -> VADBackend:
    backend_name = settings.name.strip()
    if not backend_name:
        raise RuntimeError("backend.name is required")
    if backend_name == SileroVAD.name:
        return SileroVAD(
            sample_rate=settings.sample_rate,
            frame_ms=settings.frame_ms,
            window_samples=settings.window_samples,
            history_buffer_ms=settings.history_buffer_ms,
            hangover_ms=settings.hangover_ms,
            threshold_start=settings.threshold_start,
            threshold_end=settings.threshold_end,
            model_path=settings.model_path.strip(),
            execution_provider=settings.execution_provider.strip(),
            command=settings.command.strip(),
            args=settings.args,
            timeout_sec=settings.timeout_sec,
            workspace_dir=settings.workspace_dir.strip(),
            cleanup_audio_files=settings.cleanup_audio_files,
        )
    raise RuntimeError(f"unsupported VAD backend.name: {backend_name}")
