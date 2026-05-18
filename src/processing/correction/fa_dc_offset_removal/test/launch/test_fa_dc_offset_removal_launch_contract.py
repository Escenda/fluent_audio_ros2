from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_dc_offset_removal_launch(
    config_path: Path,
) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_dc_offset_removal",
            "fa_dc_offset_removal.launch.py",
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
        PACKAGE_ROOT / "launch" / "fa_dc_offset_removal.launch.py"
    ).read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert "config/default.yaml" not in launch_text
    assert 'package="fa_dc_offset_removal"' in launch_text
    assert 'executable="fa_dc_offset_removal_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "backend.name" not in launch_text
    assert "expected.sample_rate" not in launch_text
    assert "expected.channels" not in launch_text


def test_default_launch_config_keeps_dc_offset_as_correction_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_dc_offset_removal"]["ros__parameters"]

    assert params["input_topic"] == "fa_dc_offset_removal/input"
    assert params["output_topic"] == "fa_dc_offset_removal/output"
    assert params["input_stream_id"] == "audio/sample_format/mic"
    assert params["output"]["stream_id"] == "audio/dc_offset_removed/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert "gain" not in params
    assert "filter" not in params
    assert "resample" not in params


def test_launch_fails_closed_when_input_topic_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_dc_offset_removal"]["ros__parameters"]["input_topic"] = ""
    config_path = tmp_path / "missing_input_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_dc_offset_removal_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_topic is required" in result.stdout


def test_launch_fails_closed_when_input_and_output_topics_match(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_dc_offset_removal"]["ros__parameters"]
    params["output_topic"] = params["input_topic"]
    config_path = tmp_path / "same_input_output_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_dc_offset_removal_launch(config_path)

    assert "process has died" in result.stdout
    assert "resolved input_topic and output_topic must be distinct" in result.stdout


def test_launch_fails_closed_when_resolved_input_and_output_topics_match(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_dc_offset_removal"]["ros__parameters"]
    params["input_topic"] = "audio/resolved_same"
    params["output_topic"] = "/audio/resolved_same"
    config_path = tmp_path / "resolved_same_input_output_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_dc_offset_removal_launch(config_path)

    assert "process has died" in result.stdout
    assert "resolved input_topic and output_topic must be distinct" in result.stdout


def test_launch_fails_closed_when_encoding_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_dc_offset_removal"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_dc_offset_removal_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_dc_offset_removal requires expected.encoding=FLOAT32LE" in result.stdout


def test_launch_fails_closed_when_input_stream_id_matches_topic(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_dc_offset_removal"]["ros__parameters"]
    params["input_stream_id"] = params["input_topic"]
    config_path = tmp_path / "input_stream_matches_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_dc_offset_removal_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id must be distinct from ROS topics" in result.stdout


def test_launch_fails_closed_when_output_stream_id_matches_topic(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_dc_offset_removal"]["ros__parameters"]
    params["output"]["stream_id"] = params["output_topic"]
    config_path = tmp_path / "output_stream_matches_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_dc_offset_removal_launch(config_path)

    assert "process has died" in result.stdout
    assert "output.stream_id must be distinct from ROS topics" in result.stdout


def test_launch_fails_closed_when_input_and_output_stream_ids_match(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_dc_offset_removal"]["ros__parameters"]
    params["output"]["stream_id"] = params["input_stream_id"]
    config_path = tmp_path / "same_stream_identity.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_dc_offset_removal_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id and output.stream_id must be distinct" in result.stdout
