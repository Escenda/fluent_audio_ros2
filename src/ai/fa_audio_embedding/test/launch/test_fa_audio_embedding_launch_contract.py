from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_launch_uses_explicit_config_file_contract() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_audio_embedding.launch.py").read_text(
        encoding="utf-8"
    )

    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'DeclareLaunchArgument(\n                "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n                "config_file"' in launch_text
    assert 'LaunchConfiguration("node_name")' in launch_text
    assert "parameters=[config_file]" in launch_text


def test_default_config_does_not_select_backend_worker_or_identity() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_audio_embedding"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["backend.command"] == ""
    assert params["backend.model_id"] == ""
    assert params["backend.model_path"] == ""
    assert params["backend.args"] == []
    assert params["backend.payload_encoding"] == "float32le_raw"
    assert params["embedding.dimension"] == 0
    assert params["expected_source_id"] == ""
    assert params["expected_stream_id"] == ""
    assert params["backend.timeout_sec"] > 0
