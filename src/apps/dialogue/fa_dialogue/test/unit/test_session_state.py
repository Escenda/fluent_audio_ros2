from fa_dialogue_py.session_state import (
    CompletionEvent,
    DialogueSessionState,
    MessageStamp,
    SessionStateConfig,
    WakeEvent,
)


def _state(*, allow_zero_stamp: bool = False) -> DialogueSessionState:
    return DialogueSessionState(
        SessionStateConfig(
            session_prefix="test-session-",
            wake_max_age_ms=1000,
            wake_allow_zero_stamp=allow_zero_stamp,
        )
    )


def _wake(stamp_ms: int = 9000) -> WakeEvent:
    return WakeEvent(
        detected=True,
        keyword="fluent",
        stamp=MessageStamp(sec=stamp_ms // 1000, nanosec=(stamp_ms % 1000) * 1_000_000),
    )


def test_wake_starts_deterministic_session_and_active_turn() -> None:
    state = _state()

    decision = state.handle_wake(_wake(), now_ms=9500)

    assert decision.kind == "accepted"
    assert decision.context is not None
    assert decision.context.session_id == "test-session-1"
    assert decision.context.user_turn_id == 1
    assert decision.context.active is True


def test_active_wake_replaces_current_turn_with_next_turn_id() -> None:
    state = _state()
    first = state.handle_wake(_wake(), now_ms=9500)
    second = state.handle_wake(_wake(stamp_ms=9600), now_ms=9700)

    assert first.context is not None
    assert second.context is not None
    assert second.context.session_id == first.context.session_id
    assert second.context.user_turn_id == 2
    assert second.context.active is True

    old_turn_end = state.handle_turn_end(
        CompletionEvent(
            session_id=first.context.session_id,
            user_turn_id=first.context.user_turn_id,
            terminal=True,
        )
    )
    assert old_turn_end.kind == "ignored"
    assert old_turn_end.reason == "turn_mismatch"


def test_turn_end_for_active_turn_publishes_inactive_context() -> None:
    state = _state()
    wake = state.handle_wake(_wake(), now_ms=9500)
    assert wake.context is not None

    decision = state.handle_turn_end(
        CompletionEvent(
            session_id=wake.context.session_id,
            user_turn_id=wake.context.user_turn_id,
            terminal=True,
        )
    )

    assert decision.kind == "ended"
    assert decision.context is not None
    assert decision.context.session_id == wake.context.session_id
    assert decision.context.user_turn_id == wake.context.user_turn_id
    assert decision.context.active is False


def test_turn_end_without_active_turn_is_ignored() -> None:
    state = _state()

    turn_end_decision = state.handle_turn_end(
        CompletionEvent(session_id="test-session-1", user_turn_id=1, terminal=True)
    )

    assert turn_end_decision.reason == "turn_not_active"


def test_session_and_turn_mismatch_are_ignored_without_fallback() -> None:
    state = _state()
    wake = state.handle_wake(_wake(), now_ms=9500)
    assert wake.context is not None

    wrong_session = state.handle_turn_end(
        CompletionEvent(session_id="other-session", user_turn_id=1, terminal=True)
    )
    wrong_turn = state.handle_turn_end(
        CompletionEvent(
            session_id=wake.context.session_id,
            user_turn_id=wake.context.user_turn_id + 1,
            terminal=True,
        )
    )

    assert wrong_session.reason == "session_mismatch"
    assert wrong_turn.reason == "turn_mismatch"
    assert state.active_turn_id == wake.context.user_turn_id


def test_stale_wake_is_rejected() -> None:
    state = _state()

    decision = state.handle_wake(_wake(stamp_ms=1000), now_ms=2501)

    assert decision.kind == "rejected"
    assert decision.reason == "stale_stamp"
    assert decision.context is None
    assert state.session_id == ""


def test_future_wake_is_rejected() -> None:
    state = _state()

    decision = state.handle_wake(_wake(stamp_ms=2600), now_ms=2501)

    assert decision.kind == "rejected"
    assert decision.reason == "future_stamp"
    assert decision.context is None
    assert state.session_id == ""


def test_zero_stamp_requires_explicit_config() -> None:
    rejected = _state(allow_zero_stamp=False).handle_wake(
        WakeEvent(True, "fluent", MessageStamp(sec=0, nanosec=0)),
        now_ms=2501,
    )
    accepted = _state(allow_zero_stamp=True).handle_wake(
        WakeEvent(True, "fluent", MessageStamp(sec=0, nanosec=0)),
        now_ms=2501,
    )

    assert rejected.kind == "rejected"
    assert rejected.reason == "zero_stamp_disallowed"
    assert accepted.kind == "accepted"
    assert accepted.context is not None


def test_empty_or_undetected_wake_does_not_start_session() -> None:
    state = _state()

    not_detected = state.handle_wake(
        WakeEvent(False, "fluent", MessageStamp(sec=9, nanosec=0)),
        now_ms=9500,
    )
    empty_keyword = state.handle_wake(
        WakeEvent(True, "  ", MessageStamp(sec=9, nanosec=0)),
        now_ms=9500,
    )

    assert not_detected.reason == "not_detected"
    assert empty_keyword.reason == "empty_keyword"
    assert state.session_id == ""


def test_invalid_session_config_is_rejected() -> None:
    for prefix in ("", "  ", " padded "):
        try:
            SessionStateConfig(
                session_prefix=prefix,
                wake_max_age_ms=1000,
                wake_allow_zero_stamp=False,
            )
        except ValueError:
            continue

        raise AssertionError(f"invalid session_prefix was accepted: {prefix!r}")

    try:
        SessionStateConfig(
            session_prefix="test-session-",
            wake_max_age_ms=0,
            wake_allow_zero_stamp=False,
        )
    except ValueError:
        return

    raise AssertionError("non-positive wake_max_age_ms was accepted")
