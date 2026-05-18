from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_noise_gate_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_noise_gate",
            "fa_noise_gate.launch.py",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_noise_gate.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'default_value="fa_noise_gate"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'FindPackageShare("fa_noise_gate"), "config", "default.yaml"' in launch_text
    assert 'package="fa_noise_gate"' in launch_text
    assert 'executable="fa_noise_gate_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "gate.threshold_linear" not in launch_text
    assert "gate.closed_gain_linear" not in launch_text
    assert "expected.sample_rate" not in launch_text


def test_default_launch_config_keeps_noise_gate_as_dynamics_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_noise_gate"]["ros__parameters"]

    assert params["input_topic"] == "audio/dc_offset_removed/mic"
    assert params["output_topic"] == "audio/noise_gated/mic"
    assert params["gate"]["threshold_linear"] == 0.02
    assert params["gate"]["closed_gain_linear"] == 0.0
    assert 0.0 <= params["gate"]["threshold_linear"] <= 1.0
    assert 0.0 <= params["gate"]["closed_gain_linear"] <= 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert "resample" not in params
    assert "limiter" not in params
    assert "normalize" not in params
    assert "compressor" not in params


def test_launch_fails_closed_when_threshold_is_negative(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_noise_gate"]["ros__parameters"]["gate"]["threshold_linear"] = -0.01
    config_path = tmp_path / "negative_threshold.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_noise_gate_launch(config_path)

    assert "process has died" in result.stdout
    assert "gate.threshold_linear must be finite and in [0.0, 1.0]" in result.stdout


def test_launch_fails_closed_when_closed_gain_is_too_large(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_noise_gate"]["ros__parameters"]["gate"]["closed_gain_linear"] = 1.1
    config_path = tmp_path / "closed_gain_too_large.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_noise_gate_launch(config_path)

    assert "process has died" in result.stdout
    assert "gate.closed_gain_linear must be finite and in [0.0, 1.0]" in result.stdout


def test_launch_fails_closed_when_encoding_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_noise_gate"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_noise_gate_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_noise_gate requires expected.encoding=FLOAT32LE" in result.stdout
