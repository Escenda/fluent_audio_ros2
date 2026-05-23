from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from fa_asr_py.backends.parakeet_rnnt_stream_processor import TranscriptSnapshot


class AsrProcessor(Protocol):
    def reset(self) -> None:
        ...

    def push(self, samples_float32_16k_mono: np.ndarray) -> list[TranscriptSnapshot]:
        ...

    def finish(self) -> TranscriptSnapshot:
        ...


class AsrStreamState(str, Enum):
    IDLE = "idle"
    STREAMING = "streaming"


@dataclass(frozen=True)
class AsrStreamConfig:
    sample_rate: int = 16000
    preroll_ms: int = 500

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.preroll_ms < 0:
            raise ValueError("preroll_ms must be >= 0")


@dataclass(frozen=True)
class AsrTranscriptEvent:
    text: str
    is_final: bool
    accepted_samples: int
    session_id: str
    user_turn_id: int


class AsrStreamController:
    """Thin lifecycle gate around the ASR model stream.

    Dialogue ownership stays outside this object. It only reacts to explicit
    start/stop/cancel controls and never inspects VAD, KWS, TD, or turn context.
    """

    def __init__(self, processor: AsrProcessor, config: AsrStreamConfig) -> None:
        self.processor = processor
        self.config = config
        self._preroll_samples = int(config.sample_rate * config.preroll_ms / 1000)
        self.reset()

    @property
    def state(self) -> AsrStreamState:
        return self._state

    def reset(self) -> None:
        self.processor.reset()
        self._state = AsrStreamState.IDLE
        self._preroll = np.empty(0, dtype=np.float32)
        self._session_id = ""
        self._user_turn_id = 0
        self._last_partial_text = ""

    def start(self, *, session_id: str, user_turn_id: int) -> list[AsrTranscriptEvent]:
        if not session_id:
            raise ValueError("session_id is required")
        if user_turn_id <= 0:
            raise ValueError("user_turn_id must be > 0")
        self.processor.reset()
        self._session_id = session_id
        self._user_turn_id = int(user_turn_id)
        self._last_partial_text = ""
        self._state = AsrStreamState.STREAMING
        if self._preroll.size == 0:
            return []
        events = self._events_from_snapshots(
            self.processor.push(self._preroll),
            is_final=False,
        )
        self._preroll = np.empty(0, dtype=np.float32)
        return events

    def stop(self) -> list[AsrTranscriptEvent]:
        if self._state != AsrStreamState.STREAMING:
            return []
        snapshot = self.processor.finish()
        event = self._event_from_snapshot(snapshot, is_final=True)
        self.processor.reset()
        self._state = AsrStreamState.IDLE
        self._session_id = ""
        self._user_turn_id = 0
        self._last_partial_text = ""
        return [event]

    def cancel(self) -> list[AsrTranscriptEvent]:
        if self._state == AsrStreamState.STREAMING:
            self.processor.reset()
        self._state = AsrStreamState.IDLE
        self._session_id = ""
        self._user_turn_id = 0
        self._last_partial_text = ""
        return []

    def on_audio(self, samples_float32_16k_mono: np.ndarray) -> list[AsrTranscriptEvent]:
        samples = self._validate_samples(samples_float32_16k_mono)
        if samples.size == 0:
            return []
        if self._state != AsrStreamState.STREAMING:
            self._append_preroll(samples)
            return []
        return self._events_from_snapshots(self.processor.push(samples), is_final=False)

    def _events_from_snapshots(
        self,
        snapshots: list[TranscriptSnapshot],
        *,
        is_final: bool,
    ) -> list[AsrTranscriptEvent]:
        events: list[AsrTranscriptEvent] = []
        for snapshot in snapshots:
            text = snapshot.text.strip()
            if not is_final and (not text or text == self._last_partial_text):
                continue
            if not is_final:
                self._last_partial_text = text
            events.append(self._event_from_snapshot(snapshot, is_final=is_final))
        return events

    def _event_from_snapshot(
        self,
        snapshot: TranscriptSnapshot,
        *,
        is_final: bool,
    ) -> AsrTranscriptEvent:
        return AsrTranscriptEvent(
            text=snapshot.text,
            is_final=is_final,
            accepted_samples=int(snapshot.accepted_samples),
            session_id=self._session_id,
            user_turn_id=int(self._user_turn_id),
        )

    def _append_preroll(self, samples: np.ndarray) -> None:
        if self._preroll_samples <= 0:
            self._preroll = np.empty(0, dtype=np.float32)
            return
        self._preroll = np.concatenate((self._preroll, samples))
        if self._preroll.size > self._preroll_samples:
            self._preroll = self._preroll[-self._preroll_samples :]

    @staticmethod
    def _validate_samples(samples: np.ndarray) -> np.ndarray:
        if not isinstance(samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray")
        if samples.ndim != 1:
            raise ValueError("samples must be mono 1-D float32 PCM")
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples contain non-finite values")
        return np.clip(samples, -1.0, 1.0)
