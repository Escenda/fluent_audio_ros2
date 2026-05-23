from __future__ import annotations

import pytest

from fa_vad_py.speech_activity import SpeechActivityConfig, SpeechActivityEstimator
from fa_vad_py.vad_probability_stream import VadProbabilityFrame


def _estimator() -> SpeechActivityEstimator:
    return SpeechActivityEstimator(
        SpeechActivityConfig(
            speech_threshold=0.55,
            silence_threshold=0.30,
            start_delta=0.20,
            end_delta=0.20,
            min_start_probability=0.35,
            max_end_probability=0.45,
            smoothing_alpha=0.5,
            start_consecutive_windows=2,
            end_consecutive_windows=2,
            min_speech_ms=0,
            sample_rate=16000,
        )
    )


def _frames(probabilities: list[float]) -> list[VadProbabilityFrame]:
    return [
        VadProbabilityFrame(
            probability=probability,
            window_start_sample=index * 512,
            window_end_sample=(index + 1) * 512,
        )
        for index, probability in enumerate(probabilities)
    ]


def test_hysteresis_detects_speech_start_and_end_from_probability_frames() -> None:
    estimator = _estimator()
    snapshots = [
        estimator.update(frame)
        for frame in _frames([0.05, 0.12, 0.62, 0.70, 0.68, 0.22, 0.08])
    ]

    assert [item.speech_started for item in snapshots] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert [item.speech_ended for item in snapshots] == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert snapshots[3].is_speech is True
    assert snapshots[-1].is_speech is False


def test_probability_delta_can_start_speech_below_main_threshold() -> None:
    estimator = SpeechActivityEstimator(
        SpeechActivityConfig(
            speech_threshold=0.70,
            silence_threshold=0.30,
            start_delta=0.20,
            end_delta=0.20,
            min_start_probability=0.35,
            max_end_probability=0.45,
            smoothing_alpha=0.5,
            start_consecutive_windows=1,
            end_consecutive_windows=2,
            min_speech_ms=0,
            sample_rate=16000,
        )
    )
    snapshots = [estimator.update(frame) for frame in _frames([0.04, 0.36])]

    assert snapshots[1].speech_started is True


def test_invalid_config_rejects_reversed_hysteresis() -> None:
    with pytest.raises(ValueError, match="silence_threshold"):
        SpeechActivityConfig(speech_threshold=0.2, silence_threshold=0.5)
