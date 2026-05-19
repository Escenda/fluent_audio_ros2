from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_declares_minimal_turn_context_contract() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_dialogue"]["ros__parameters"]

    assert params["wake_word_topic"] == "voice/wake_word"
    assert params["asr_result_topic"] == "voice/asr/result"
    assert params["turn_end_topic"] == "voice/turn_end"
    assert params["turn_context_topic"] == "conversation/turn_context"
    assert params["session_prefix"] == "dialogue-session-"
    assert params["wake.max_age_ms"] > 0
    assert params["wake.allow_zero_stamp"] is False
    assert params["wake.qos.depth"] == 10
    assert params["wake.qos.reliable"] is True
    assert params["asr.qos.depth"] == 10
    assert params["asr.qos.reliable"] is True
    assert params["turn_end.qos.depth"] == 10
    assert params["turn_end.qos.reliable"] is True
    assert params["turn_context.qos.depth"] == 10
    assert params["turn_context.qos.reliable"] is True
