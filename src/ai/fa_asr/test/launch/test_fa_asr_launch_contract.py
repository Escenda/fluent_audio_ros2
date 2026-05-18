from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_launch_uses_explicit_config_file_contract() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_asr.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'FindPackageShare("fa_asr"), "config", "default.yaml"' in launch_text
    assert "parameters=[config_file]" in launch_text


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
    assert params["expected_source_id"] == ""
    assert params["expected_stream_id"] == ""
    assert params["backend.timeout_sec"] > 0
