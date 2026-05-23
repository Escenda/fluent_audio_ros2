from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fa_vad_py.vad_probability_stream import VadProbabilityStream


@dataclass
class FakeVadBackend:
    probabilities: list[float]
    sample_rate: int = 16000
    window_size_samples: int = 512
    name: str = "fake_vad"

    def reset(self) -> None:
        self.index = 0

    def predict_probability(self, samples: np.ndarray) -> float:
        assert samples.size == self.window_size_samples
        probability = self.probabilities[self.index]
        self.index += 1
        return probability


def test_probability_stream_buffers_partial_windows_until_enough_samples_arrive() -> None:
    stream = VadProbabilityStream(FakeVadBackend([0.80]))
    half_window = stream.backend.window_size_samples // 2

    first = stream.push(np.zeros(half_window, dtype=np.float32))
    second = stream.push(np.zeros(half_window, dtype=np.float32))

    assert first == []
    assert len(second) == 1
    assert second[0].probability == 0.80
    assert second[0].window_start_sample == 0
    assert second[0].window_end_sample == stream.backend.window_size_samples


def test_probability_stream_emits_one_frame_per_model_window() -> None:
    stream = VadProbabilityStream(FakeVadBackend([0.1, 0.7, 0.2]))
    samples = np.zeros(stream.backend.window_size_samples * 3, dtype=np.float32)

    frames = stream.push(samples)

    assert [frame.probability for frame in frames] == [0.1, 0.7, 0.2]
    assert [frame.window_start_sample for frame in frames] == [0, 512, 1024]
    assert [frame.window_end_sample for frame in frames] == [512, 1024, 1536]
