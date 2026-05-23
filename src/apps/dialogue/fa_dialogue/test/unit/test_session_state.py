from fa_dialogue_py.session_state import (
    DialogueTurnConfig,
    DialogueTurnController,
    MessageStamp,
    TurnEndCandidate,
    VoiceActivityEvent,
    WakeEvent,
)


def _state(
    *,
    allow_zero_stamp: bool = False,
    min_active_ms: int = 0,
) -> DialogueTurnController:
    return DialogueTurnController(
        DialogueTurnConfig(
            session_prefix="test-session-",
            wake_max_age_ms=1000,
            wake_allow_zero_stamp=allow_zero_stamp,
            min_active_ms=min_active_ms,
            no_speech_timeout_ms=6000,
            quiet_candidate_ms=1200,
            td_threshold=0.65,
            fallback_quiet_ms=3500,
            max_active_ms=30000,
        )
    )


def _wake(stamp_ms: int = 9000) -> WakeEvent:
    return WakeEvent(
        detected=True,
        keyword="fluent",
        stamp=MessageStamp(sec=stamp_ms // 1000, nanosec=(stamp_ms % 1000) * 1_000_000),
    )


def _start_turn_with_speech(state: DialogueTurnController, *, now_ms: int = 9000):
    wake = state.handle_wake(_wake(stamp_ms=now_ms), now_ms=now_ms)
    context = wake.contexts[0]
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=now_ms + 100,
    )
    return wake, context


def _request_turn_end(
    state: DialogueTurnController,
    *,
    wake_ms: int = 9000,
    quiet_start_ms: int = 10100,
):
    wake, context = _start_turn_with_speech(state, now_ms=wake_ms)
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=False, speech_started=False, speech_ended=True),
        now_ms=quiet_start_ms,
    )
    request_decision = state.tick(now_ms=quiet_start_ms + 1200)
    request = request_decision.turn_end_requests[0]
    return wake, context, request_decision, request


def test_wake_starts_turn_context_and_asr_control() -> None:
    state = _state()

    decision = state.handle_wake(_wake(), now_ms=9500)

    assert decision.kind == "accepted"
    assert decision.contexts[0].session_id == "test-session-1"
    assert decision.contexts[0].user_turn_id == 1
    assert decision.contexts[0].active is True
    assert decision.asr_controls[0].action == "start"
    assert decision.asr_controls[0].reason == "wake"


def test_wake_during_active_turn_is_ignored() -> None:
    state = _state()
    first = state.handle_wake(_wake(), now_ms=9500)
    second = state.handle_wake(_wake(stamp_ms=9600), now_ms=9700)

    assert first.kind == "accepted"
    assert second.kind == "ignored"
    assert second.reason == "turn_active"
    assert state.active_turn_id == 1


def test_td_candidate_before_dialogue_request_is_ignored() -> None:
    state = _state()
    wake, context = _start_turn_with_speech(state)

    decision = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id, 1, True, 0.9),
        now_ms=9800,
    )

    assert wake.kind == "accepted"
    assert decision.kind == "ignored"
    assert decision.reason == "no_pending_td_request"
    assert state.active_turn_id == context.user_turn_id


def test_quiet_candidate_requests_turn_detection_without_stopping_asr() -> None:
    state = _state()
    wake, context = _start_turn_with_speech(state, now_ms=9000)
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=False, speech_started=False, speech_ended=True),
        now_ms=10100,
    )

    too_early = state.tick(now_ms=11299)
    requested = state.tick(now_ms=11300)

    assert wake.kind == "accepted"
    assert too_early.kind == "ignored"
    assert requested.kind == "accepted"
    assert requested.reason == "turn_end_requested"
    assert not requested.asr_controls
    assert requested.turn_end_requests[0].session_id == context.session_id
    assert requested.turn_end_requests[0].user_turn_id == context.user_turn_id
    assert requested.turn_end_requests[0].quiet_ms == 1200
    assert state.active_turn_id == context.user_turn_id


def test_td_result_above_threshold_stops_matching_request() -> None:
    state = _state()
    _, context, _, request = _request_turn_end(state)

    ended = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id, request.request_id, True, 0.8),
        now_ms=11350,
    )

    assert ended.kind == "ended"
    assert ended.reason == "td_end"
    assert ended.asr_controls[0].action == "stop"
    assert ended.contexts[0].active is False


def test_td_result_below_threshold_waits_for_quiet_fallback() -> None:
    state = _state()
    _, context, _, request = _request_turn_end(state)

    weak = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id, request.request_id, True, 0.5),
        now_ms=11350,
    )
    waiting = state.tick(now_ms=13599)
    fallback = state.tick(now_ms=13600)

    assert weak.kind == "accepted"
    assert weak.reason == "td_not_end"
    assert waiting.kind == "ignored"
    assert fallback.kind == "ended"
    assert fallback.reason == "quiet_fallback"


def test_speech_restart_cancels_pending_td_request() -> None:
    state = _state()
    _, context, _, request = _request_turn_end(state)

    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=11400,
    )
    stale = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id, request.request_id, True, 0.9),
        now_ms=11450,
    )
    decision = state.tick(now_ms=12000)

    assert stale.kind == "ignored"
    assert stale.reason == "no_pending_td_request"
    assert decision.kind == "ignored"
    assert state.active_turn_id == context.user_turn_id


def test_wake_without_followup_speech_stops_after_timeout() -> None:
    state = _state()

    wake = state.handle_wake(_wake(), now_ms=9000)
    waiting = state.tick(now_ms=14999)
    ended = state.tick(now_ms=15000)

    assert wake.kind == "accepted"
    assert waiting.reason == "waiting_for_speech"
    assert ended.kind == "ended"
    assert ended.reason == "no_speech_timeout"


def test_min_active_window_blocks_early_td_request() -> None:
    state = _state(min_active_ms=3000)
    _, context = _start_turn_with_speech(state, now_ms=9000)
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=False, speech_started=False, speech_ended=True),
        now_ms=9300,
    )

    too_early = state.tick(now_ms=11999)
    requested = state.tick(now_ms=12000)

    assert too_early.kind == "ignored"
    assert state.active_turn_id == context.user_turn_id
    assert requested.kind == "accepted"
    assert requested.reason == "turn_end_requested"


def test_request_id_mismatch_is_ignored() -> None:
    state = _state()
    _, context, _, request = _request_turn_end(state)

    wrong = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id, request.request_id + 1, True, 0.9),
        now_ms=11350,
    )

    assert wrong.reason == "td_request_mismatch"
    assert state.active_turn_id == context.user_turn_id


def test_session_and_turn_mismatch_are_ignored() -> None:
    state = _state()
    _, context, _, request = _request_turn_end(state)

    wrong_session = state.handle_turn_end_candidate(
        TurnEndCandidate("other-session", context.user_turn_id, request.request_id, True, 0.9),
        now_ms=11350,
    )
    wrong_turn = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id + 1, request.request_id, True, 0.9),
        now_ms=11350,
    )

    assert wrong_session.reason == "session_mismatch"
    assert wrong_turn.reason == "turn_mismatch"
    assert state.active_turn_id == context.user_turn_id


def test_max_active_timeout_stops_runaway_turn() -> None:
    state = DialogueTurnController(
        DialogueTurnConfig(
            session_prefix="test-session-",
            wake_max_age_ms=1000,
            wake_allow_zero_stamp=False,
            min_active_ms=0,
            no_speech_timeout_ms=6000,
            quiet_candidate_ms=1200,
            td_threshold=0.65,
            fallback_quiet_ms=3500,
            max_active_ms=5000,
        )
    )
    wake, _ = _start_turn_with_speech(state, now_ms=9000)

    ended = state.tick(now_ms=14000)

    assert wake.kind == "accepted"
    assert ended.kind == "ended"
    assert ended.reason == "max_active_timeout"


def test_stale_future_and_zero_stamp_wake_validation() -> None:
    stale = _state().handle_wake(_wake(stamp_ms=1000), now_ms=2501)
    future = _state().handle_wake(_wake(stamp_ms=2600), now_ms=2501)
    zero_rejected = _state(allow_zero_stamp=False).handle_wake(
        WakeEvent(True, "fluent", MessageStamp(sec=0, nanosec=0)),
        now_ms=2501,
    )
    zero_accepted = _state(allow_zero_stamp=True).handle_wake(
        WakeEvent(True, "fluent", MessageStamp(sec=0, nanosec=0)),
        now_ms=2501,
    )

    assert stale.reason == "stale_stamp"
    assert future.reason == "future_stamp"
    assert zero_rejected.reason == "zero_stamp_disallowed"
    assert zero_accepted.kind == "accepted"


def test_invalid_config_is_rejected() -> None:
    invalid_values = [
        {"session_prefix": ""},
        {"session_prefix": " padded "},
        {"wake_max_age_ms": 0},
        {"min_active_ms": -1},
        {"no_speech_timeout_ms": 0},
        {"no_speech_timeout_ms": 2999},
        {"quiet_candidate_ms": 0},
        {"td_threshold": -0.1},
        {"td_threshold": 1.1},
        {"fallback_quiet_ms": 0},
        {"fallback_quiet_ms": 1199},
        {"max_active_ms": 0},
        {"max_active_ms": 2999},
    ]
    for overrides in invalid_values:
        values = {
            "session_prefix": "test-session-",
            "wake_max_age_ms": 1000,
            "wake_allow_zero_stamp": False,
            "min_active_ms": 3000,
            "no_speech_timeout_ms": 6000,
            "quiet_candidate_ms": 1200,
            "td_threshold": 0.65,
            "fallback_quiet_ms": 3500,
            "max_active_ms": 30000,
        }
        values.update(overrides)
        try:
            DialogueTurnConfig(**values)
        except ValueError:
            continue
        raise AssertionError(f"invalid config was accepted: {overrides}")
