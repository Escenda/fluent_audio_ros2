from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fa_asr_py.audio_frame import AudioFrameContract, audio_frame_to_float32_mono


def _frame(**overrides: object) -> SimpleNamespace:
    values = {
        "source_id": "mic",
        "stream_id": "audio/preprocessed/mono16k",
        "sample_rate": 16000,
        "channels": 1,
        "encoding": "FLOAT32LE",
        "bit_depth": 32,
        "layout": "interleaved",
        "data": np.array([0.0, 0.5, -0.5], dtype="<f4").tobytes(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _contract() -> AudioFrameContract:
    return AudioFrameContract(
        source_id="mic",
        stream_id="audio/preprocessed/mono16k",
        sample_rate=16000,
    )


def test_audio_frame_to_float32_mono_accepts_expected_contract() -> None:
    samples = audio_frame_to_float32_mono(_frame(), _contract())

    np.testing.assert_allclose(samples, np.array([0.0, 0.5, -0.5], dtype=np.float32))


def test_audio_frame_to_float32_mono_rejects_wrong_stream_id() -> None:
    with pytest.raises(ValueError, match="stream_id"):
        audio_frame_to_float32_mono(_frame(stream_id="audio/raw/mic"), _contract())


def test_audio_frame_to_float32_mono_rejects_out_of_range_samples() -> None:
    data = np.array([0.0, 1.5], dtype="<f4").tobytes()

    with pytest.raises(ValueError, match="normalized"):
        audio_frame_to_float32_mono(_frame(data=data), _contract())
