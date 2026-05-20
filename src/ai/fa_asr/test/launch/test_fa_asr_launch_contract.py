from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_does_not_select_backend_or_worker_implicitly() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_asr"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["backend.command"] == ""
    assert params["backend.model"] == ""
    assert params["backend.model_path"] == ""
    assert params["backend.openai_realtime.api_key_env"] == ""
    assert params["backend.openai_transcriptions.api_key_env"] == ""
    assert params["backend.args"] == []
    assert params["backend.result_format"] == ""
    assert params["expected_source_id"] == ""
    assert params["expected_stream_id"] == ""
    assert params["backend.timeout_sec"] > 0
    assert params["audio.qos.depth"] == 20
    assert params["audio.qos.reliable"] is False
    assert params["vad.qos.depth"] == 50
    assert params["vad.qos.reliable"] is False
    assert params["turn_context.qos.depth"] == 10
    assert params["turn_context.qos.reliable"] is True
    assert params["result.qos.depth"] == 10
    assert params["result.qos.reliable"] is True
