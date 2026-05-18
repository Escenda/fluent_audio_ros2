from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]
LAUNCH_TIMEOUT_CODE = 124


def _run_fa_aec_linear_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    command = [
        ros2,
        "launch",
        "fa_aec_linear",
        "fa_aec_linear.launch.py",
        "node_name:=fa_aec_linear",
        f"config_file:={config_path}",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, _ = process.communicate(timeout=8)
        return subprocess.CompletedProcess(command, process.returncode, stdout, None)
    except subprocess.TimeoutExpired:
        process.terminate()
        stdout, _ = process.communicate(timeout=2)
        return subprocess.CompletedProcess(command, LAUNCH_TIMEOUT_CODE, stdout, None)


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_aec_linear.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert "config/default.yaml" not in launch_text
    assert 'package="fa_aec_linear"' in launch_text
    assert 'executable="fa_aec_linear_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "cancel_gain" not in launch_text
    assert "expected_sample_rate" not in launch_text
    assert "expected.encoding" not in launch_text


def test_default_launch_config_keeps_aec_linear_as_correction_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_aec_linear"]["ros__parameters"]

    assert params["mic_topic"] == "audio/resample16k/mic"
    assert params["ref_topic"] == "audio/resample16k/ref"
    assert params["output_topic"] == "audio/aec_linear/frame"
    assert params["mic_stream_id"] == "audio/mic/resample16k"
    assert params["ref_stream_id"] == "audio/ref/resample16k"
    assert params["output"]["stream_id"] == "audio/aec_linear/output"
    assert params["expected_sample_rate"] == 16000
    assert params["expected_channels"] == 1
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert params["reference_failure_policy"] == "drop"
    assert "resample" not in params
    assert "limiter" not in params
    assert "normalize" not in params


def test_launch_fails_closed_when_expected_encoding_contract_is_wrong(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_aec_linear"]["ros__parameters"]["expected"]["encoding"] = "PCM32LE"
    config["fa_aec_linear"]["ros__parameters"]["expected"]["bit_depth"] = 32
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_aec_linear_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert "expected encoding/bit_depth must be PCM16LE/16 or FLOAT32LE/32" in result.stdout


def test_launch_fails_closed_when_resolved_topics_match(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_aec_linear"]["ros__parameters"]
    params["mic_topic"] = "audio/aec_same"
    params["output_topic"] = "/audio/aec_same"
    config_path = tmp_path / "same_resolved_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_aec_linear_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert "resolved mic_topic and output_topic must be distinct" in result.stdout


def test_launch_fails_closed_when_stream_id_matches_topic(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_aec_linear"]["ros__parameters"]
    params["mic_stream_id"] = "audio/resample16k/mic"
    config_path = tmp_path / "stream_id_matches_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_aec_linear_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert "must be distinct from ROS topics" in result.stdout


def test_launch_accepts_default_config(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "default.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_aec_linear_launch(config_path)

    assert result.returncode == LAUNCH_TIMEOUT_CODE
    assert "Starting FA AEC Linear node" in result.stdout
    assert "Exception:" not in result.stdout
