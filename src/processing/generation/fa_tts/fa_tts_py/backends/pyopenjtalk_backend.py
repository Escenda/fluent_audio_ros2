from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from fa_tts_py.backends.base import SynthesizedAudio


class PyOpenJTalkBackend:
    name = "pyopenjtalk"

    def __init__(self, *, openjtalk_dict_dir: str) -> None:
        dictionary_dir = Path(openjtalk_dict_dir).expanduser()
        if not openjtalk_dict_dir.strip():
            raise RuntimeError("backend.openjtalk_dict_dir is required for pyopenjtalk")
        if not dictionary_dir.is_dir():
            raise RuntimeError(f"backend.openjtalk_dict_dir is not a directory: {dictionary_dir}")
        os.environ["OPEN_JTALK_DICT_DIR"] = str(dictionary_dir)
        os.environ["TQDM_DISABLE"] = "1"
        # Backend runtime import is intentionally scoped to the selected backend.
        import pyopenjtalk

        self._runtime = pyopenjtalk

    def synthesize(self, text: str, voice_id: str) -> SynthesizedAudio:
        options: dict[str, str] = {}
        if voice_id:
            options["voice"] = voice_id
        wav, sample_rate = self._runtime.tts(text, **options)
        waveform = np.asarray(wav, dtype=np.float32)
        if waveform.ndim != 1:
            raise RuntimeError("pyopenjtalk waveform must be mono")
        if not np.all(np.isfinite(waveform)):
            raise RuntimeError("pyopenjtalk waveform contains non-finite samples")
        if np.any(waveform < -1.0) or np.any(waveform > 1.0):
            raise RuntimeError("pyopenjtalk waveform must be normalized to [-1.0, 1.0]")
        return SynthesizedAudio(
            audio_bytes=waveform.astype(np.float32, copy=False).tobytes(),
            encoding="FLOAT32LE",
            sample_rate=int(sample_rate),
            channels=1,
            bit_depth=32,
        )
