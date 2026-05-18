from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_declick_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_declick",
            "fa_declick.launch.py",
            "node_name:=fa_declick",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_declick.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert "config/default.yaml" not in launch_text
    assert 'package="fa_declick"' in launch_text
    assert 'executable="fa_declick_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "threshold.delta" not in launch_text
    assert "expected.sample_rate" not in launch_text


def test_launch_fails_closed_when_input_topic_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_declick"]["ros__parameters"]["input_topic"] = ""
    config_path = tmp_path / "missing_input_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_declick_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_topic is required" in result.stdout


def test_launch_fails_closed_when_input_stream_id_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_declick"]["ros__parameters"]["input_stream_id"] = ""
    config_path = tmp_path / "missing_input_stream_id.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_declick_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id is required" in result.stdout


def test_launch_fails_closed_when_output_stream_id_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_declick"]["ros__parameters"]["output"]["stream_id"] = ""
    config_path = tmp_path / "missing_output_stream_id.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_declick_launch(config_path)

    assert "process has died" in result.stdout
    assert "output.stream_id is required" in result.stdout


def test_launch_fails_closed_when_resolved_input_and_output_topics_match(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_declick"]["ros__parameters"]
    params["input_topic"] = "audio/declick_same"
    params["output_topic"] = "/audio/declick_same"
    config_path = tmp_path / "same_resolved_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_declick_launch(config_path)

    assert "process has died" in result.stdout
    assert "resolved input_topic and output_topic must be distinct" in result.stdout


def test_launch_fails_closed_when_stream_identity_matches_resolved_topic(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_declick"]["ros__parameters"]
    params["input_topic"] = "audio/declick_input"
    params["input_stream_id"] = "/audio/declick_input"
    config_path = tmp_path / "stream_matches_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_declick_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id must be distinct from ROS topics" in result.stdout


def test_launch_fails_closed_when_input_and_output_stream_identity_match(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_declick"]["ros__parameters"]
    params["input_stream_id"] = "same_stream"
    params["output"]["stream_id"] = "same_stream"
    config_path = tmp_path / "same_stream_identity.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_declick_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id and output.stream_id must be distinct" in result.stdout


def test_launch_fails_closed_when_threshold_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_declick"]["ros__parameters"]["threshold"]["delta"] = 0.0
    config_path = tmp_path / "bad_threshold.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_declick_launch(config_path)

    assert "process has died" in result.stdout
    assert "threshold.delta must be finite and in (0.0, 2.0]" in result.stdout


def test_launch_fails_closed_when_expected_encoding_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_declick"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "bad_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_declick_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_declick requires expected.encoding=FLOAT32LE" in result.stdout
