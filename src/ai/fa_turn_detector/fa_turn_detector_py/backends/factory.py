from __future__ import annotations

from dataclasses import dataclass

from fa_turn_detector_py.backends.base import TurnDetectorBackend
from fa_turn_detector_py.backends.smart_turn_onnx import SmartTurnOnnxBackend


@dataclass(frozen=True)
class TurnDetectorBackendSettings:
    name: str
    model_path: str
    threshold: float
    execution_provider: str
    command: str
    args: tuple[str, ...]
    health_args: tuple[str, ...]
    timeout_sec: float
    workspace_dir: str
    cleanup_audio_files: bool


def build_turn_detector_backend(
    settings: TurnDetectorBackendSettings,
) -> TurnDetectorBackend:
    backend_name = settings.name.strip()
    if not backend_name:
        raise RuntimeError("backend.name is required")
    if backend_name == SmartTurnOnnxBackend.name:
        return SmartTurnOnnxBackend(
            model_path=settings.model_path.strip(),
            threshold=settings.threshold,
            execution_provider=settings.execution_provider.strip(),
            command=settings.command.strip(),
            args=settings.args,
            health_args=settings.health_args,
            timeout_sec=settings.timeout_sec,
            workspace_dir=settings.workspace_dir.strip(),
            cleanup_audio_files=settings.cleanup_audio_files,
        )
    raise RuntimeError(f"unsupported turn detector backend.name: {backend_name}")
