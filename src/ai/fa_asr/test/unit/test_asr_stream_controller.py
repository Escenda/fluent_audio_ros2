from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fa_asr_py.asr_stream_controller import (
    AsrStreamConfig,
    AsrStreamController,
    AsrStreamState,
)
from fa_asr_py.backends.parakeet_rnnt_stream_processor import TranscriptSnapshot


@dataclass
class FakeAsrProcessor:
    pushed: list[np.ndarray] = field(default_factory=list)
    reset_count: int = 0
    finished: bool = False

    def reset(self) -> None:
        self.reset_count += 1
        self.finished = False

    def push(self, samples_float32_16k_mono: np.ndarray) -> list[TranscriptSnapshot]:
        samples = samples_float32_16k_mono.copy()
        self.pushed.append(samples)
        return [
            TranscriptSnapshot(
                text=f"text-{len(self.pushed)}",
                accepted_samples=sum(item.size for item in self.pushed),
                complete=False,
            )
        ]

    def finish(self) -> TranscriptSnapshot:
        self.finished = True
        return TranscriptSnapshot(
            text="final text",
            accepted_samples=sum(item.size for item in self.pushed),
            complete=True,
        )


def _controller() -> AsrStreamController:
    return AsrStreamController(
        FakeAsrProcessor(),
        AsrStreamConfig(sample_rate=16000, preroll_ms=20),
    )


def test_audio_before_start_is_kept_as_preroll_and_pushed_on_start() -> None:
    controller = _controller()
    processor = controller.processor
    controller.on_audio(np.ones(640, dtype=np.float32))

    events = controller.start(session_id="session-a", user_turn_id=1)

    assert controller.state == AsrStreamState.STREAMING
    assert len(events) == 1
    assert processor.pushed[0].size == 320
    np.testing.assert_allclose(processor.pushed[0], np.ones(320, dtype=np.float32))
    assert events[0].session_id == "session-a"
    assert events[0].user_turn_id == 1


def test_streaming_audio_produces_partial_and_stop_produces_final() -> None:
    controller = _controller()
    controller.start(session_id="session-b", user_turn_id=3)

    partial = controller.on_audio(np.zeros(512, dtype=np.float32))
    final = controller.stop()

    assert partial[0].text == "text-1"
    assert partial[0].is_final is False
    assert final[0].text == "final text"
    assert final[0].is_final is True
    assert final[0].session_id == "session-b"
    assert final[0].user_turn_id == 3
    assert controller.state == AsrStreamState.IDLE


def test_cancel_resets_without_final_transcript() -> None:
    controller = _controller()
    processor = controller.processor
    controller.start(session_id="session-c", user_turn_id=5)
    controller.on_audio(np.zeros(512, dtype=np.float32))

    events = controller.cancel()

    assert events == []
    assert processor.finished is False
    assert controller.state == AsrStreamState.IDLE


def test_rejects_start_without_session_identity() -> None:
    controller = _controller()

    try:
        controller.start(session_id="", user_turn_id=1)
    except ValueError as exc:
        assert "session_id" in str(exc)
    else:
        raise AssertionError("empty session_id was accepted")

    try:
        controller.start(session_id="session", user_turn_id=0)
    except ValueError as exc:
        assert "user_turn_id" in str(exc)
    else:
        raise AssertionError("non-positive user_turn_id was accepted")
