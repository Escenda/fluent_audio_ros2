from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_sample_format_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_sample_format",
            "fa_sample_format.launch.py",
            "node_name:=fa_sample_format",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (
        PACKAGE_ROOT / "launch" / "fa_sample_format.launch.py"
    ).read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
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
    assert params["input_stream_id"] == "audio/raw/mic"
    assert params["output"]["stream_id"] == "audio/float32/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input"]["encoding"] == "PCM16LE"
    assert params["input"]["bit_depth"] == 16
    assert params["output"]["encoding"] == "FLOAT32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["layout"] == "interleaved"


def test_launch_fails_closed_when_conversion_is_unsupported(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_sample_format"]["ros__parameters"]["input"]["encoding"] = "MULAW"
    config_path = tmp_path / "unsupported_conversion.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_sample_format_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_sample_format supports only PCM16LE/16" in result.stdout


def test_launch_fails_closed_when_input_topic_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_sample_format"]["ros__parameters"]["input_topic"] = ""
    config_path = tmp_path / "missing_input_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_sample_format_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_topic is required" in result.stdout
