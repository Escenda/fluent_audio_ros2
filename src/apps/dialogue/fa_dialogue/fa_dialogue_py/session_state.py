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
    min_active_ms: int = 3000
    no_speech_timeout_ms: int = 6000
    quiet_candidate_ms: int = 1200
    td_threshold: float = 0.65
    fallback_quiet_ms: int = 3500
    max_active_ms: int = 30000

    def __post_init__(self) -> None:
        if not self.session_prefix.strip():
            raise ValueError("session_prefix is required")
        if self.session_prefix != self.session_prefix.strip():
            raise ValueError("session_prefix must not have surrounding whitespace")
        if self.wake_max_age_ms <= 0:
            raise ValueError("wake_max_age_ms must be positive")
        if self.min_active_ms < 0:
            raise ValueError("min_active_ms must be >= 0")
        if self.no_speech_timeout_ms <= 0:
            raise ValueError("no_speech_timeout_ms must be positive")
        if self.no_speech_timeout_ms < self.min_active_ms:
            raise ValueError("no_speech_timeout_ms must be >= min_active_ms")
        if self.quiet_candidate_ms <= 0:
            raise ValueError("quiet_candidate_ms must be positive")
        if self.td_threshold < 0.0 or self.td_threshold > 1.0:
            raise ValueError("td_threshold must be in [0.0, 1.0]")
        if self.fallback_quiet_ms <= 0:
            raise ValueError("fallback_quiet_ms must be positive")
        if self.fallback_quiet_ms < self.quiet_candidate_ms:
            raise ValueError("fallback_quiet_ms must be >= quiet_candidate_ms")
        if self.max_active_ms <= 0:
            raise ValueError("max_active_ms must be positive")
        if self.max_active_ms < self.min_active_ms:
            raise ValueError("max_active_ms must be >= min_active_ms")


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
    request_id: int
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
class TurnEndRequestCommand:
    session_id: str
    user_turn_id: int
    request_id: int
    quiet_ms: int


@dataclass(frozen=True)
class DialogueDecision:
    kind: DecisionKind
    reason: str
    contexts: tuple[TurnContextSnapshot, ...] = field(default_factory=tuple)
    asr_controls: tuple[AsrControlCommand, ...] = field(default_factory=tuple)
    turn_end_requests: tuple[TurnEndRequestCommand, ...] = field(default_factory=tuple)


class DialogueTurnController:
    """Owns dialogue turn lifecycle and ASR control decisions."""

    def __init__(self, config: DialogueTurnConfig) -> None:
        self._config = config
        self._next_session_sequence = 1
        self._next_turn_id = 1
        self._next_turn_end_request_id = 1
        self._session_id = ""
        self._active_turn_id = 0
        self._turn_started_at_ms: int | None = None
        self._has_user_speech = False
        self._latest_vad_is_speech = False
        self._quiet_started_at_ms: int | None = None
        self._pending_td_request_id = 0
        self._td_checked_quiet_started_at_ms: int | None = None

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
        self._has_user_speech = False
        self._quiet_started_at_ms = None
        self._pending_td_request_id = 0
        self._td_checked_quiet_started_at_ms = None

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

        if event.speech_started or event.is_speech:
            self._has_user_speech = True
            self._quiet_started_at_ms = None
            self._pending_td_request_id = 0
            self._td_checked_quiet_started_at_ms = None
            return DialogueDecision("accepted", "speech_active")

        if not self._has_user_speech:
            return DialogueDecision("ignored", "waiting_for_speech")

        if self._quiet_started_at_ms is None:
            self._quiet_started_at_ms = now_ms
            self._pending_td_request_id = 0
            self._td_checked_quiet_started_at_ms = None
            return DialogueDecision("accepted", "quiet_started")

        return DialogueDecision("ignored", "quiet_continues")

    def handle_turn_end_candidate(
        self,
        event: TurnEndCandidate,
        *,
        now_ms: int,
    ) -> DialogueDecision:
        mismatch = self._completion_mismatch_reason(event.session_id, event.user_turn_id)
        if mismatch is not None:
            return DialogueDecision("ignored", mismatch)
        if self._pending_td_request_id == 0:
            return DialogueDecision("ignored", "no_pending_td_request")
        if int(event.request_id) != self._pending_td_request_id:
            return DialogueDecision("ignored", "td_request_mismatch")
        if self._quiet_started_at_ms is None:
            self._pending_td_request_id = 0
            return DialogueDecision("ignored", "speech_resumed")

        self._pending_td_request_id = 0
        self._td_checked_quiet_started_at_ms = self._quiet_started_at_ms
        if (
            float(event.probability) >= self._config.td_threshold
            and self._turn_age_ms(now_ms) >= self._config.min_active_ms
        ):
            return self._end_turn("td_end")
        return DialogueDecision("accepted", "td_not_end")

    def tick(self, *, now_ms: int) -> DialogueDecision:
        if not self.active:
            return DialogueDecision("ignored", "turn_not_active")

        turn_age_ms = self._turn_age_ms(now_ms)
        if turn_age_ms >= self._config.max_active_ms:
            return self._end_turn("max_active_timeout")

        if not self._has_user_speech:
            if turn_age_ms >= self._config.no_speech_timeout_ms:
                return self._end_turn("no_speech_timeout")
            return DialogueDecision("ignored", "waiting_for_speech")

        if self._quiet_started_at_ms is None:
            return DialogueDecision("ignored", "speech_active")

        quiet_ms = now_ms - self._quiet_started_at_ms
        if (
            quiet_ms >= self._config.quiet_candidate_ms
            and turn_age_ms >= self._config.min_active_ms
            and self._pending_td_request_id == 0
            and self._td_checked_quiet_started_at_ms != self._quiet_started_at_ms
        ):
            return self._request_turn_end(quiet_ms=quiet_ms)

        if (
            quiet_ms >= self._config.fallback_quiet_ms
            and turn_age_ms >= self._config.min_active_ms
        ):
            return self._end_turn("quiet_fallback")
        return DialogueDecision("ignored", "waiting_for_end")

    def _request_turn_end(self, *, quiet_ms: int) -> DialogueDecision:
        request_id = self._next_turn_end_request_id
        self._next_turn_end_request_id += 1
        self._pending_td_request_id = request_id
        return DialogueDecision(
            "accepted",
            "turn_end_requested",
            turn_end_requests=(
                TurnEndRequestCommand(
                    session_id=self._session_id,
                    user_turn_id=self._active_turn_id,
                    request_id=request_id,
                    quiet_ms=max(0, quiet_ms),
                ),
            ),
        )

    def _end_turn(self, reason: str) -> DialogueDecision:
        session_id = self._session_id
        turn_id = self._active_turn_id
        self._active_turn_id = 0
        self._turn_started_at_ms = None
        self._has_user_speech = False
        self._quiet_started_at_ms = None
        self._pending_td_request_id = 0
        self._td_checked_quiet_started_at_ms = None
        return DialogueDecision(
            "ended",
            reason,
            contexts=(TurnContextSnapshot(session_id, turn_id, False),),
            asr_controls=(AsrControlCommand("stop", session_id, turn_id, reason),),
        )

    def _turn_age_ms(self, now_ms: int) -> int:
        if self._turn_started_at_ms is None:
            return 0
        return max(0, now_ms - self._turn_started_at_ms)

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
