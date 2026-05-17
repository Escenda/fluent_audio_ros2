from __future__ import annotations

from fa_tts_py.backends.base import TextToSpeechBackend
from fa_tts_py.backends.pyopenjtalk_backend import PyOpenJTalkBackend


def build_tts_backend(name: str, *, openjtalk_dict_dir: str) -> TextToSpeechBackend:
    backend_name = name.strip()
    if not backend_name:
        raise RuntimeError("backend.name is required")
    if backend_name == PyOpenJTalkBackend.name:
        return PyOpenJTalkBackend(openjtalk_dict_dir=openjtalk_dict_dir)
    raise RuntimeError(f"unsupported fa_tts backend.name: {backend_name}")
