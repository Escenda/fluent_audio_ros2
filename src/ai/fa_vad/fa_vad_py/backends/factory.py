from __future__ import annotations

from fa_vad_py.backends.base import VadBackendSettings, VadProbabilityBackend
from fa_vad_py.backends.silero_vad import SileroVadOnnxBackend


def build_vad_backend(settings: VadBackendSettings) -> VadProbabilityBackend:
    if settings.name == "silero_vad":
        return SileroVadOnnxBackend(settings)
    raise RuntimeError(f"unsupported VAD backend.name: {settings.name}")
