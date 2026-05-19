from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_requires_explicit_backend_and_external_worker_command() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    assert "fa_vad" in config
    assert "fa_vad_node" not in config
    params = config["fa_vad"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["input_topic"] == "audio/frame"
    assert params["input_stream_id"] == "audio/raw/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["backend.command"] == ""
    assert params["backend.frame_ms"] == 20
    assert params["backend.window_samples"] == 512
    assert params["backend.history_buffer_ms"] == 200
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""
    assert params["expected_source_id"] == ""
    assert params["backend.timeout_sec"] > 0
    assert params["backend.workspace_dir"]
    assert params["backend.cleanup_audio_files"] is True
    assert params["qos.depth"] == 10
    assert params["qos.reliable"] is False

    rendered_args = " ".join(params["backend.args"])
    for placeholder in (
        "{audio}",
        "{model}",
        "{provider}",
        "{sample_rate}",
        "{window_samples}",
    ):
        assert placeholder in rendered_args
