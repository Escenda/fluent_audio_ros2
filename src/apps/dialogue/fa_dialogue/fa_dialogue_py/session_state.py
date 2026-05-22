from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


WakeDecisionKind = Literal["accepted", "ignored", "rejected"]
WakeDecisionReason = Literal[
    "accepted",
    "not_detected",
    "empty_keyword",
    "zero_stamp_disallowed",
    "future_stamp",
    "stale_stamp",
]
CompletionDecisionKind = Literal["ended", "ignored"]
CompletionDecisionReason = Literal[
    "ended",
    "turn_not_active",
    "session_mismatch",
    "turn_mismatch",
    "not_terminal",
]


@dataclass(frozen=True)
class MessageStamp:
    sec: int
    nanosec: int

    def is_zero(self) -> bool:
        return self.sec == 0 and self.nanosec == 0

    def to_milliseconds(self) -> int:
        return (self.sec * 1000) + (self.nanosec // 1_000_000)


@dataclass(frozen=True)
class SessionStateConfig:
    session_prefix: str
    wake_max_age_ms: int
    wake_allow_zero_stamp: bool

    def __post_init__(self) -> None:
        if not self.session_prefix.strip():
            raise ValueError("session_prefix is required")
        if self.session_prefix != self.session_prefix.strip():
            raise ValueError("session_prefix must not have surrounding whitespace")
        if self.wake_max_age_ms <= 0:
            raise ValueError("wake_max_age_ms must be positive")


@dataclass(frozen=True)
class WakeEvent:
    detected: bool
    keyword: str
    stamp: MessageStamp


@dataclass(frozen=True)
class CompletionEvent:
    session_id: str
    user_turn_id: int
    terminal: bool


@dataclass(frozen=True)
class TurnContextSnapshot:
    session_id: str
    user_turn_id: int
    active: bool


@dataclass(frozen=True)
class WakeDecision:
    kind: WakeDecisionKind
    reason: WakeDecisionReason
    context: TurnContextSnapshot | None


@dataclass(frozen=True)
class CompletionDecision:
    kind: CompletionDecisionKind
    reason: CompletionDecisionReason
    context: TurnContextSnapshot | None


class DialogueSessionState:
    def __init__(self, config: SessionStateConfig) -> None:
        self._config = config
        self._next_session_sequence = 1
        self._session_id = ""
        self._next_turn_id = 1
        self._active_turn_id = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def active_turn_id(self) -> int:
        return self._active_turn_id

    def handle_wake(self, event: WakeEvent, *, now_ms: int) -> WakeDecision:
        if not event.detected:
            return WakeDecision("ignored", "not_detected", None)
        if not event.keyword.strip():
            return WakeDecision("ignored", "empty_keyword", None)

        freshness = self._validate_wake_freshness(event.stamp, now_ms=now_ms)
        if freshness != "accepted":
            return WakeDecision("rejected", freshness, None)

        if not self._session_id:
            self._session_id = self._build_session_id()
            self._next_session_sequence += 1

        turn_id = self._next_turn_id
        self._next_turn_id += 1
        self._active_turn_id = turn_id

        return WakeDecision(
            "accepted",
            "accepted",
            TurnContextSnapshot(
                session_id=self._session_id,
                user_turn_id=turn_id,
                active=True,
            ),
        )

    def handle_turn_end(self, event: CompletionEvent) -> CompletionDecision:
        return self._handle_completion(event)

    def _handle_completion(self, event: CompletionEvent) -> CompletionDecision:
        if not event.terminal:
            return CompletionDecision("ignored", "not_terminal", None)
        if self._active_turn_id == 0:
            return CompletionDecision("ignored", "turn_not_active", None)
        if event.session_id != self._session_id:
            return CompletionDecision("ignored", "session_mismatch", None)
        if event.user_turn_id != self._active_turn_id:
            return CompletionDecision("ignored", "turn_mismatch", None)

        ended_turn_id = self._active_turn_id
        self._active_turn_id = 0
        return CompletionDecision(
            "ended",
            "ended",
            TurnContextSnapshot(
                session_id=self._session_id,
                user_turn_id=ended_turn_id,
                active=False,
            ),
        )

    def _validate_wake_freshness(
        self,
        stamp: MessageStamp,
        *,
        now_ms: int,
    ) -> WakeDecisionReason:
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
