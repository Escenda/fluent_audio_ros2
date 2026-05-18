from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]
LAUNCH_TIMEOUT_CODE = 124


def _run_fa_aec_nn_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    command = [
        ros2,
        "launch",
        "fa_aec_nn",
        "fa_aec_nn.launch.py",
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
    launch_text = (PACKAGE_ROOT / "launch" / "fa_aec_nn.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert "config/default.yaml" not in launch_text
    assert 'package="fa_aec_nn"' in launch_text
    assert 'executable="fa_aec_nn_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "backend.name" not in launch_text
    assert "expected_sample_rate" not in launch_text
    assert "expected_channels" not in launch_text


def test_default_launch_config_requires_explicit_backend() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_aec_nn"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["input_topic"] == "audio/aec_linear/frame"
    assert params["output_topic"] == "audio/aec/frame"
    assert params["expected_sample_rate"] == 16000
    assert params["expected_channels"] == 1
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert "resample" not in params
    assert "gain" not in params
    assert "aec_linear" not in params


def test_launch_fails_closed_when_backend_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "missing_backend.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_aec_nn_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert "backend.name is required" in result.stdout


def test_launch_fails_closed_when_backend_is_unknown(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_aec_nn"]["ros__parameters"]["backend.name"] = "unknown"
    config_path = tmp_path / "unknown_backend.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_aec_nn_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert "backend.name must be passthrough" in result.stdout


def test_launch_accepts_explicit_passthrough_config(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_aec_nn"]["ros__parameters"]["backend.name"] = "passthrough"
    config_path = tmp_path / "passthrough.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_aec_nn_launch(config_path)

    assert result.returncode == LAUNCH_TIMEOUT_CODE
    assert "backend.name is required" not in result.stdout
    assert "backend.name must be passthrough" not in result.stdout
    assert "Starting FA AEC NN node" in result.stdout
    assert "Exception:" not in result.stdout
