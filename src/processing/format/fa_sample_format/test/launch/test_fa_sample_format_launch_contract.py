from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (
        PACKAGE_ROOT / "launch" / "fa_sample_format.launch.py"
    ).read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert (
        'FindPackageShare("fa_sample_format"), "config", "default.yaml"'
        in launch_text
    )
    assert 'package="fa_sample_format"' in launch_text
    assert 'executable="fa_sample_format_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "input.encoding" not in launch_text
    assert "output.encoding" not in launch_text
    assert "backend.name" not in launch_text


def test_default_launch_config_keeps_sample_format_as_explicit_format_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_sample_format"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"
    assert params["output_topic"] == "audio/sample_format/mic"
    assert params["input"]["encoding"] == "PCM16LE"
    assert params["input"]["bit_depth"] == 16
    assert params["output"]["encoding"] == "FLOAT32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["layout"] == "interleaved"
