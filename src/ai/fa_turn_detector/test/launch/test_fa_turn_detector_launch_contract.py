from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_requires_explicit_backend_and_external_worker_boundary() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_turn_detector"]["ros__parameters"]

    assert params["audio_topic"] == "audio/frame"
    assert params["expected_stream_id"] == "audio/raw/mic"
    assert params["audio_topic"] != params["expected_stream_id"]
    assert params["expected_source_id"] == ""
    assert params["control.default_enabled"] is False
    assert params["control.inputs"] == ["speech_control"]
    assert params["control.speech_control.action"] == "topic"
    assert params["control.speech_control.topic"] == "voice/vad_state"
    assert params["control.speech_control.msg_type"] == "fa_interfaces/msg/VadState"
    assert params["control.speech_control.source_id"] == ""
    assert params["control.speech_control.stream_id"] == "audio/raw/mic"
    assert params["control.speech_control.active_field"] == "is_speech"
    assert params["control.speech_control.start_field"] == "start"
    assert params["control.speech_control.end_field"] == "end"
    assert params["control.speech_control.close_on"] == "end_or_active_falling"
    assert params["control.speech_control.qos.depth"] == 10
    assert params["control.speech_control.qos.reliable"] is False
    assert params["backend.name"] == ""
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""
    assert params["backend.command"] == ""
    assert params["backend.args"] == []
    assert params["backend.health_args"] == []
    assert params["backend.timeout_sec"] > 0
    assert params["backend.workspace_dir"]
    assert params["backend.cleanup_audio_files"] is True
    assert params["backend.threshold"] == 0.5
    assert params["audio.qos.depth"] == 10
    assert params["audio.qos.reliable"] is False
    assert params["turn_context.qos.depth"] == 10
    assert params["turn_context.qos.reliable"] is True
    assert params["output.qos.depth"] == 10
    assert params["output.qos.reliable"] is True
