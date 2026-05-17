from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from fa_tts_py.backends.base import SynthesizedAudio


class PyOpenJTalkBackend:
    name = "pyopenjtalk"

    def __init__(self) -> None:
        os.environ.setdefault("OPEN_JTALK_DICT_DIR", str(Path.home() / ".pyopenjtalk"))
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

        if waveform.size:
            abs_max = float(np.max(np.abs(waveform)))
            if abs_max > 1.0:
                waveform /= 32768.0
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm = (waveform * 32767.0).astype(np.int16)
        return SynthesizedAudio(
            audio_bytes=pcm.tobytes(),
            sample_rate=int(sample_rate),
            channels=1,
            bit_depth=16,
        )
