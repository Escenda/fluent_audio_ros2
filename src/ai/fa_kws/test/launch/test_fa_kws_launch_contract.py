from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_requires_explicit_backend_and_kws_inputs() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_kws"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["backend.execution_provider"] == ""
    assert params["backend.command"] == ""
    assert params["backend.args"] == []
    assert params["backend.health_args"] == []
    assert params["backend.timeout_sec"] > 0
    assert params["backend.workspace_dir"]
    assert params["backend.cleanup_audio_files"] is True
    assert params["model.encoder"] == ""
    assert params["model.decoder"] == ""
    assert params["model.joiner"] == ""
    assert params["model.tokens"] == ""
    assert params["kws.keywords_file"] == ""
    assert params["audio_topic"] == "audio/frame"
    assert params["expected_stream_id"] == "audio/raw/mic"
    assert params["audio_topic"] != params["expected_stream_id"]
    assert params["expected_source_id"] == ""
    assert params["vad.probability_gate"] == 0.35
    assert params["vad.max_age_ms"] > 0
    assert params["audio.qos.depth"] > 0
    assert params["audio.qos.reliable"] is False
    assert params["vad.qos.depth"] > 0
    assert params["vad.qos.reliable"] is False
    assert params["output.qos.depth"] > 0
    assert params["output.qos.reliable"] is False
    assert "dump_audio.enable" not in params
    assert "dump_audio.path" not in params
