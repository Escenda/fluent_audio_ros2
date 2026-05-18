from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_resample.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'FindPackageShare("fa_resample"), "config", "default.yaml"' in launch_text
    assert 'package="fa_resample"' in launch_text
    assert 'executable="fa_resample_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "target_sample_rate" not in launch_text
    assert "backend.name" not in launch_text


def test_default_launch_config_keeps_resample_as_explicit_format_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_resample"]["ros__parameters"]

    assert params["target_sample_rate"] == 16000
    assert params["input"]["encoding"] == "FLOAT32LE"
    assert params["input"]["bit_depth"] == 32
    assert params["input"]["layout"] == "interleaved"
    assert params["output"]["encoding"] == "FLOAT32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["mic"]["enabled"] is True
    assert params["mic"]["input_topic"] == "audio/frame"
    assert params["mic"]["output_topic"] == "audio/resample16k/mic"
