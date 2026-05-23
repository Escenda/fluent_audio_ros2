from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from fa_asr_py.backends.parakeet_rnnt_stream_processor import (
    ParakeetRnntStreamProcessor,
    ParakeetStreamConfig,
)


def _normalize_transcript(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower()).strip()


def _generate_flite_audio(text: str, *, voice: str) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    if importlib.util.find_spec("soundfile") is None:
        pytest.skip("soundfile is not installed")

    import soundfile as sf

    output_path = Path("/tmp") / f"fa_asr_parakeet_{voice}.wav"
    result = subprocess.run(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"flite=text={text}:voice={voice}",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg flite source is not available: {result.stderr.strip()}")

    audio, sample_rate = sf.read(str(output_path), dtype="float32", always_2d=False)
    if sample_rate != 16000:
        pytest.fail(f"generated audio sample_rate mismatch: {sample_rate}")
    if audio.ndim != 1:
        audio = audio.mean(axis=1).astype(np.float32)
    audio = audio.astype(np.float32, copy=False)
    return np.concatenate(
        (
            np.zeros(3200, dtype=np.float32),
            audio,
            np.zeros(3200, dtype=np.float32),
        )
    )


def test_real_parakeet_model_stream_processor_smoke() -> None:
    model_path = os.environ.get("FA_ASR_PARAKEET_MODEL_PATH", "").strip()
    if not model_path:
        pytest.skip("FA_ASR_PARAKEET_MODEL_PATH is not set")
    if os.environ.get("FA_ASR_RUN_REAL_MODEL", "") != "1":
        pytest.skip("FA_ASR_RUN_REAL_MODEL=1 is required for real-model smoke")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    if importlib.util.find_spec("nemo") is None:
        pytest.skip("nemo is not installed")
    if importlib.util.find_spec("omegaconf") is None:
        pytest.skip("omegaconf is not installed")
    if not Path(model_path).is_file():
        pytest.fail(f"FA_ASR_PARAKEET_MODEL_PATH does not exist: {model_path}")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for 1.1B Parakeet smoke test")

    processor = ParakeetRnntStreamProcessor(
        ParakeetStreamConfig(
            model_path=model_path,
            device="cuda",
            compute_dtype="bfloat16",
            sample_rate=16000,
            left_context_secs=0.64,
            chunk_secs=0.64,
            right_context_secs=0.32,
        )
    )

    input_text = "good morning this is a longer test of the streaming transcription system"
    audio = _generate_flite_audio(input_text, voice="slt")
    step = processor.context_samples.chunk // 2
    snapshots = []
    for offset in range(0, audio.size, step):
        snapshots.extend(processor.push(audio[offset : offset + step]))
    final = processor.finish()
    normalized_final = _normalize_transcript(final.text)
    partial_texts = [snapshot.text for snapshot in snapshots if snapshot.text.strip()]

    print(f"input_text={input_text!r}")
    print(f"partial_transcripts={partial_texts!r}")
    print(f"final_transcript={final.text!r}")

    assert snapshots
    assert partial_texts
    assert final.complete is True
    assert final.accepted_samples >= audio.size
    for word in ("good", "morning", "longer", "streaming", "transcription", "system"):
        assert word in normalized_final, f"word={word!r} transcript={final.text!r}"
