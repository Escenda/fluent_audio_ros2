from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_limiter_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_limiter",
            "fa_limiter.launch.py",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_limiter.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'FindPackageShare("fa_limiter"), "config", "default.yaml"' in launch_text
    assert 'package="fa_limiter"' in launch_text
    assert 'executable="fa_limiter_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "threshold.linear" not in launch_text
    assert "expected.sample_rate" not in launch_text
    assert "expected.channels" not in launch_text


def test_default_launch_config_keeps_limiter_as_dynamics_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_limiter"]["ros__parameters"]

    assert params["input_topic"] == "audio/gain/mic"
    assert params["output_topic"] == "audio/limit/mic"
    assert params["threshold"]["linear"] == 1.0
    assert 0.0 < params["threshold"]["linear"] <= 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert "resample" not in params
    assert "normalize" not in params
    assert "gain" not in params


def test_launch_fails_closed_when_threshold_is_zero(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_limiter"]["ros__parameters"]["threshold"]["linear"] = 0.0
    config_path = tmp_path / "zero_threshold.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_limiter_launch(config_path)

    assert "process has died" in result.stdout
    assert "threshold.linear must be finite and in (0.0, 1.0]" in result.stdout


def test_launch_fails_closed_when_threshold_exceeds_unity(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_limiter"]["ros__parameters"]["threshold"]["linear"] = 1.1
    config_path = tmp_path / "too_high_threshold.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_limiter_launch(config_path)

    assert "process has died" in result.stdout
    assert "threshold.linear must be finite and in (0.0, 1.0]" in result.stdout


def test_launch_fails_closed_when_encoding_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_limiter"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_limiter_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_limiter requires expected.encoding=FLOAT32LE" in result.stdout
