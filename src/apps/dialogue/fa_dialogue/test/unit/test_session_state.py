from fa_dialogue_py.session_state import (
    DialogueTurnConfig,
    DialogueTurnController,
    MessageStamp,
    TurnEndCandidate,
    VoiceActivityEvent,
    WakeEvent,
)


def _state(*, allow_zero_stamp: bool = False, min_listen_ms: int = 0) -> DialogueTurnController:
    return DialogueTurnController(
        DialogueTurnConfig(
            session_prefix="test-session-",
            wake_max_age_ms=1000,
            wake_allow_zero_stamp=allow_zero_stamp,
            min_turn_ms=1000,
            min_listen_ms=min_listen_ms,
            no_speech_timeout_ms=3000,
            td_min_silence_ms=300,
            vad_fallback_silence_ms=1500,
        )
    )


def _wake(stamp_ms: int = 9000) -> WakeEvent:
    return WakeEvent(
        detected=True,
        keyword="fluent",
        stamp=MessageStamp(sec=stamp_ms // 1000, nanosec=(stamp_ms % 1000) * 1_000_000),
    )


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


def test_td_candidate_before_vad_speech_end_is_ignored() -> None:
    state = _state()
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=9400,
    )
    wake = state.handle_wake(_wake(), now_ms=9500)
    context = wake.contexts[0]

    decision = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id, True, 0.8),
        now_ms=9800,
    )

    assert decision.kind == "ignored"
    assert decision.reason == "no_speech_end_candidate"
    assert state.active_turn_id == context.user_turn_id


def test_td_candidate_after_speech_end_stops_after_min_silence() -> None:
    state = _state()
    wake = state.handle_wake(_wake(), now_ms=9000)
    context = wake.contexts[0]
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=9100,
    )
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=False, speech_started=False, speech_ended=True),
        now_ms=10100,
    )

    pending = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id, True, 0.8),
        now_ms=10150,
    )
    ended = state.tick(now_ms=10400)

    assert pending.kind == "accepted"
    assert pending.reason == "td_pending"
    assert ended.kind == "ended"
    assert ended.reason == "td_end"
    assert ended.asr_controls[0].action == "stop"
    assert ended.contexts[0].active is False


def test_speech_restart_cancels_pending_td_end() -> None:
    state = _state()
    wake = state.handle_wake(_wake(), now_ms=9000)
    context = wake.contexts[0]
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=9100,
    )
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=False, speech_started=False, speech_ended=True),
        now_ms=10100,
    )
    state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id, True, 0.8),
        now_ms=10150,
    )

    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=10200,
    )
    decision = state.tick(now_ms=10600)

    assert decision.kind == "ignored"
    assert state.active_turn_id == context.user_turn_id


def test_wake_after_prior_vad_silence_waits_for_followup_speech() -> None:
    state = _state()
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=False, speech_started=False, speech_ended=True),
        now_ms=8900,
    )

    wake = state.handle_wake(_wake(), now_ms=9000)
    waiting = state.tick(now_ms=10500)
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=10600,
    )
    still_open = state.tick(now_ms=10850)

    assert wake.kind == "accepted"
    assert waiting.kind == "ignored"
    assert waiting.reason == "waiting_for_speech"
    assert still_open.kind == "ignored"
    assert state.active_turn_id == wake.contexts[0].user_turn_id


def test_wake_without_followup_speech_stops_after_timeout() -> None:
    state = _state()

    wake = state.handle_wake(_wake(), now_ms=9000)
    waiting = state.tick(now_ms=11900)
    ended = state.tick(now_ms=12000)

    assert wake.kind == "accepted"
    assert waiting.reason == "waiting_for_speech"
    assert ended.kind == "ended"
    assert ended.reason == "no_speech_timeout"


def test_min_turn_duration_is_measured_from_followup_speech_start() -> None:
    state = _state()
    wake = state.handle_wake(_wake(), now_ms=9000)
    context = wake.contexts[0]
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=10400,
    )
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=False, speech_started=False, speech_ended=True),
        now_ms=10500,
    )

    pending = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id, True, 0.8),
        now_ms=10800,
    )
    too_early = state.tick(now_ms=11399)

    assert pending.kind == "accepted"
    assert pending.reason == "td_pending"
    assert too_early.kind == "ignored"
    assert state.active_turn_id == wake.contexts[0].user_turn_id

    ended = state.tick(now_ms=11400)

    assert ended.kind == "ended"
    assert ended.reason == "td_end"




def test_min_listen_window_blocks_early_td_stop() -> None:
    state = _state(min_listen_ms=3000)
    wake = state.handle_wake(_wake(), now_ms=9000)
    context = wake.contexts[0]
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=9100,
    )
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=False, speech_started=False, speech_ended=True),
        now_ms=9200,
    )

    pending = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id, True, 0.8),
        now_ms=9500,
    )
    too_early = state.tick(now_ms=11999)

    assert pending.kind == "accepted"
    assert too_early.kind == "ignored"
    assert state.active_turn_id == context.user_turn_id

    ended = state.tick(now_ms=12000)

    assert ended.kind == "ended"
    assert ended.reason == "td_end"


def test_vad_fallback_stops_without_td_candidate() -> None:
    state = _state()
    wake = state.handle_wake(_wake(), now_ms=9000)
    context = wake.contexts[0]
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=9100,
    )
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=False, speech_started=False, speech_ended=True),
        now_ms=10100,
    )

    decision = state.tick(now_ms=11600)

    assert decision.kind == "ended"
    assert decision.reason == "vad_fallback"
    assert decision.asr_controls[0].session_id == context.session_id
    assert decision.asr_controls[0].user_turn_id == context.user_turn_id


def test_session_and_turn_mismatch_are_ignored() -> None:
    state = _state()
    wake = state.handle_wake(_wake(), now_ms=9500)
    context = wake.contexts[0]
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=True, speech_started=True, speech_ended=False),
        now_ms=9600,
    )
    state.handle_voice_activity(
        VoiceActivityEvent(is_speech=False, speech_started=False, speech_ended=True),
        now_ms=10100,
    )

    wrong_session = state.handle_turn_end_candidate(
        TurnEndCandidate("other-session", context.user_turn_id, True, 0.9),
        now_ms=10400,
    )
    wrong_turn = state.handle_turn_end_candidate(
        TurnEndCandidate(context.session_id, context.user_turn_id + 1, True, 0.9),
        now_ms=10400,
    )

    assert wrong_session.reason == "session_mismatch"
    assert wrong_turn.reason == "turn_mismatch"
    assert state.active_turn_id == context.user_turn_id


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
        {"min_turn_ms": -1},
        {"min_listen_ms": -1},
        {"no_speech_timeout_ms": 0},
        {"min_listen_ms": 3001},
        {"td_min_silence_ms": -1},
        {"vad_fallback_silence_ms": 0},
    ]
    for overrides in invalid_values:
        values = {
            "session_prefix": "test-session-",
            "wake_max_age_ms": 1000,
            "wake_allow_zero_stamp": False,
            "min_turn_ms": 1000,
            "min_listen_ms": 0,
            "no_speech_timeout_ms": 3000,
            "td_min_silence_ms": 300,
            "vad_fallback_silence_ms": 1500,
        }
        values.update(overrides)
        try:
            DialogueTurnConfig(**values)
        except ValueError:
            continue
        raise AssertionError(f"invalid config was accepted: {overrides}")
