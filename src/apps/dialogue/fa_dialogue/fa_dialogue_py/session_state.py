from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AsrControlAction = Literal["start", "stop", "cancel"]
DecisionKind = Literal["accepted", "ignored", "rejected", "ended"]


@dataclass(frozen=True)
class MessageStamp:
    sec: int
    nanosec: int

    def is_zero(self) -> bool:
        return self.sec == 0 and self.nanosec == 0

    def to_milliseconds(self) -> int:
        return (self.sec * 1000) + (self.nanosec // 1_000_000)


@dataclass(frozen=True)
class DialogueTurnConfig:
    session_prefix: str
    wake_max_age_ms: int
    wake_allow_zero_stamp: bool
    min_turn_ms: int = 1200
    min_listen_ms: int = 3000
    no_speech_timeout_ms: int = 5000
    td_min_silence_ms: int = 300
    vad_fallback_silence_ms: int = 1800

    def __post_init__(self) -> None:
        if not self.session_prefix.strip():
            raise ValueError("session_prefix is required")
        if self.session_prefix != self.session_prefix.strip():
            raise ValueError("session_prefix must not have surrounding whitespace")
        if self.wake_max_age_ms <= 0:
            raise ValueError("wake_max_age_ms must be positive")
        if self.min_turn_ms < 0:
            raise ValueError("min_turn_ms must be >= 0")
        if self.min_listen_ms < 0:
            raise ValueError("min_listen_ms must be >= 0")
        if self.no_speech_timeout_ms <= 0:
            raise ValueError("no_speech_timeout_ms must be positive")
        if self.no_speech_timeout_ms < self.min_listen_ms:
            raise ValueError("no_speech_timeout_ms must be >= min_listen_ms")
        if self.td_min_silence_ms < 0:
            raise ValueError("td_min_silence_ms must be >= 0")
        if self.vad_fallback_silence_ms <= 0:
            raise ValueError("vad_fallback_silence_ms must be positive")


@dataclass(frozen=True)
class WakeEvent:
    detected: bool
    keyword: str
    stamp: MessageStamp


@dataclass(frozen=True)
class VoiceActivityEvent:
    is_speech: bool
    speech_started: bool
    speech_ended: bool


@dataclass(frozen=True)
class TurnEndCandidate:
    session_id: str
    user_turn_id: int
    terminal: bool
    probability: float


@dataclass(frozen=True)
class TurnContextSnapshot:
    session_id: str
    user_turn_id: int
    active: bool


@dataclass(frozen=True)
class AsrControlCommand:
    action: AsrControlAction
    session_id: str
    user_turn_id: int
    reason: str


@dataclass(frozen=True)
class DialogueDecision:
    kind: DecisionKind
    reason: str
    contexts: tuple[TurnContextSnapshot, ...] = field(default_factory=tuple)
    asr_controls: tuple[AsrControlCommand, ...] = field(default_factory=tuple)


class DialogueTurnController:
    """Owns dialogue turn lifecycle and ASR control decisions."""

    def __init__(self, config: DialogueTurnConfig) -> None:
        self._config = config
        self._next_session_sequence = 1
        self._next_turn_id = 1
        self._session_id = ""
        self._active_turn_id = 0
        self._turn_started_at_ms: int | None = None
        self._speech_started_at_ms: int | None = None
        self._latest_vad_is_speech = False
        self._vad_is_speech = False
        self._speech_ended_at_ms: int | None = None
        self._pending_td_end = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def active_turn_id(self) -> int:
        return self._active_turn_id

    @property
    def active(self) -> bool:
        return self._active_turn_id > 0

    def handle_wake(self, event: WakeEvent, *, now_ms: int) -> DialogueDecision:
        if not event.detected:
            return DialogueDecision("ignored", "not_detected")
        if not event.keyword.strip():
            return DialogueDecision("ignored", "empty_keyword")
        freshness = self._validate_wake_freshness(event.stamp, now_ms=now_ms)
        if freshness != "accepted":
            return DialogueDecision("rejected", freshness)
        if self.active:
            return DialogueDecision("ignored", "turn_active")

        if not self._session_id:
            self._session_id = self._build_session_id()
            self._next_session_sequence += 1

        turn_id = self._next_turn_id
        self._next_turn_id += 1
        self._active_turn_id = turn_id
        self._turn_started_at_ms = now_ms
        self._vad_is_speech = self._latest_vad_is_speech
        self._speech_started_at_ms = now_ms if self._latest_vad_is_speech else None
        self._speech_ended_at_ms = None
        self._pending_td_end = False

        return DialogueDecision(
            "accepted",
            "wake",
            contexts=(TurnContextSnapshot(self._session_id, turn_id, True),),
            asr_controls=(AsrControlCommand("start", self._session_id, turn_id, "wake"),),
        )

    def handle_voice_activity(
        self,
        event: VoiceActivityEvent,
        *,
        now_ms: int,
    ) -> DialogueDecision:
        self._latest_vad_is_speech = bool(event.is_speech)
        if not self.active:
            return DialogueDecision("ignored", "turn_not_active")

        self._vad_is_speech = bool(event.is_speech)
        if event.speech_started or event.is_speech:
            if self._speech_started_at_ms is None:
                self._speech_started_at_ms = now_ms
            self._speech_ended_at_ms = None
            self._pending_td_end = False
            return DialogueDecision("accepted", "speech_active")

        if event.speech_ended:
            if self._speech_started_at_ms is None:
                return DialogueDecision("ignored", "speech_end_before_speech_start")
            self._speech_ended_at_ms = now_ms
            self._pending_td_end = False
            return DialogueDecision("accepted", "speech_end_candidate")

        return DialogueDecision("ignored", "no_transition")

    def handle_turn_end_candidate(
        self,
        event: TurnEndCandidate,
        *,
        now_ms: int,
    ) -> DialogueDecision:
        mismatch = self._completion_mismatch_reason(event.session_id, event.user_turn_id)
        if mismatch is not None:
            return DialogueDecision("ignored", mismatch)
        if not event.terminal:
            return DialogueDecision("ignored", "not_terminal")
        if self._speech_ended_at_ms is None:
            return DialogueDecision("ignored", "no_speech_end_candidate")

        self._pending_td_end = True
        if self._can_end_by_td(now_ms):
            return self._end_turn("td_end")
        return DialogueDecision("accepted", "td_pending")

    def tick(self, *, now_ms: int) -> DialogueDecision:
        if not self.active:
            return DialogueDecision("ignored", "turn_not_active")
        if self._speech_started_at_ms is None:
            if (
                self._turn_age_ms(now_ms) >= self._config.no_speech_timeout_ms
                and not self._latest_vad_is_speech
            ):
                return self._end_turn("no_speech_timeout")
            return DialogueDecision("ignored", "waiting_for_speech")
        if self._speech_ended_at_ms is None:
            return DialogueDecision("ignored", "no_pending_end")
        if self._pending_td_end and self._can_end_by_td(now_ms):
            return self._end_turn("td_end")
        silence_ms = now_ms - self._speech_ended_at_ms
        if (
            silence_ms >= self._config.vad_fallback_silence_ms
            and self._speech_age_ms(now_ms) >= self._config.min_turn_ms
            and self._turn_age_ms(now_ms) >= self._config.min_listen_ms
        ):
            return self._end_turn("vad_fallback")
        return DialogueDecision("ignored", "waiting_for_end")

    def _end_turn(self, reason: str) -> DialogueDecision:
        session_id = self._session_id
        turn_id = self._active_turn_id
        self._active_turn_id = 0
        self._turn_started_at_ms = None
        self._speech_started_at_ms = None
        self._vad_is_speech = False
        self._speech_ended_at_ms = None
        self._pending_td_end = False
        return DialogueDecision(
            "ended",
            reason,
            contexts=(TurnContextSnapshot(session_id, turn_id, False),),
            asr_controls=(AsrControlCommand("stop", session_id, turn_id, reason),),
        )

    def _can_end_by_td(self, now_ms: int) -> bool:
        if self._speech_ended_at_ms is None:
            return False
        silence_ms = now_ms - self._speech_ended_at_ms
        return (
            silence_ms >= self._config.td_min_silence_ms
            and self._speech_age_ms(now_ms) >= self._config.min_turn_ms
            and self._turn_age_ms(now_ms) >= self._config.min_listen_ms
        )

    def _turn_age_ms(self, now_ms: int) -> int:
        if self._turn_started_at_ms is None:
            return 0
        return max(0, now_ms - self._turn_started_at_ms)

    def _speech_age_ms(self, now_ms: int) -> int:
        if self._speech_started_at_ms is None:
            return 0
        return max(0, now_ms - self._speech_started_at_ms)

    def _completion_mismatch_reason(
        self,
        session_id: str,
        user_turn_id: int,
    ) -> str | None:
        if not self.active:
            return "turn_not_active"
        if session_id != self._session_id:
            return "session_mismatch"
        if int(user_turn_id) != self._active_turn_id:
            return "turn_mismatch"
        return None

    def _validate_wake_freshness(
        self,
        stamp: MessageStamp,
        *,
        now_ms: int,
    ) -> str:
        if stamp.is_zero():
            if self._config.wake_allow_zero_stamp:
                return "accepted"
            return "zero_stamp_disallowed"

        age_ms = now_ms - stamp.to_milliseconds()
        if age_ms < 0:
            return "future_stamp"
        if age_ms > self._config.wake_max_age_ms:
            return "stale_stamp"
        return "accepted"

    def _build_session_id(self) -> str:
        return f"{self._config.session_prefix}{self._next_session_sequence}"
