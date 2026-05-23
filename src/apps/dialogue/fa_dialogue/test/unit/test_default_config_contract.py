from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_declares_dialogue_turn_controller_contract() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_dialogue"]["ros__parameters"]

    assert params["wake_word_topic"] == "voice/wake_word"
    assert params["voice_activity_topic"] == "voice/activity"
    assert params["turn_end_topic"] == "voice/turn_end"
    assert params["turn_context_topic"] == "conversation/turn_context"
    assert params["asr_control_topic"] == "voice/asr_control"
    assert params["session_prefix"] == "dialogue-session-"
    assert params["wake.max_age_ms"] > 0
    assert params["wake.allow_zero_stamp"] is False
    assert params["turn.min_duration_ms"] >= 0
    assert params["turn.min_listen_ms"] >= 3000
    assert params["turn.no_speech_timeout_ms"] >= params["turn.min_listen_ms"]
    assert params["turn.td_min_silence_ms"] >= 0
    assert params["turn.vad_fallback_silence_ms"] > 0
    assert params["turn.tick_period_ms"] > 0
    assert params["wake.qos.depth"] == 10
    assert params["wake.qos.reliable"] is True
    assert params["voice_activity.qos.depth"] == 10
    assert params["voice_activity.qos.reliable"] is False
    assert params["turn_end.qos.depth"] == 10
    assert params["turn_end.qos.reliable"] is True
    assert params["turn_context.qos.depth"] == 10
    assert params["turn_context.qos.reliable"] is True
    assert params["asr_control.qos.depth"] == 10
    assert params["asr_control.qos.reliable"] is True
