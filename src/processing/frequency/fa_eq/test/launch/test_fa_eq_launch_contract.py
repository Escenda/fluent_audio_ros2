from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_eq_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_eq",
            "fa_eq.launch.py",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_eq.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'package="fa_eq"' in launch_text
    assert 'executable="fa_eq_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "low.cutoff_hz" not in launch_text
    assert "high.cutoff_hz" not in launch_text
    assert "gains.low_db" not in launch_text
    assert "expected.sample_rate" not in launch_text


def test_default_launch_config_keeps_eq_as_frequency_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_eq"]["ros__parameters"]

    assert params["input_topic"] == "audio/sample_format/mic"
    assert params["output_topic"] == "audio/eq/mic"
    assert params["low"]["cutoff_hz"] == 250.0
    assert params["high"]["cutoff_hz"] == 4000.0
    assert 0.0 < params["low"]["cutoff_hz"] < params["high"]["cutoff_hz"]
    assert params["high"]["cutoff_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["gains"]["low_db"] == 0.0
    assert params["gains"]["mid_db"] == 0.0
    assert params["gains"]["high_db"] == 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert "resample" not in params
    assert "normalize" not in params
    assert "limiter" not in params
    assert "denoise" not in params


def test_launch_fails_closed_when_low_cutoff_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_eq"]["ros__parameters"]["low"]["cutoff_hz"] = 0.0
    config_path = tmp_path / "invalid_low_cutoff.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_eq_launch(config_path)

    assert "process has died" in result.stdout
    assert "low.cutoff_hz must be finite and > 0.0" in result.stdout


def test_launch_fails_closed_when_high_cutoff_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_eq"]["ros__parameters"]["high"]["cutoff_hz"] = 250.0
    config_path = tmp_path / "invalid_high_cutoff.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_eq_launch(config_path)

    assert "process has died" in result.stdout
    assert (
        "high.cutoff_hz must be finite, > low.cutoff_hz, and < expected.sample_rate / 2.0"
        in result.stdout
    )


def test_launch_fails_closed_when_gain_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_eq"]["ros__parameters"]["gains"]["low_db"] = float("nan")
    config_path = tmp_path / "invalid_gain.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_eq_launch(config_path)

    assert "process has died" in result.stdout
    assert "gains.*_db must be finite" in result.stdout


def test_launch_fails_closed_when_encoding_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_eq"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_eq_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_eq requires expected.encoding=FLOAT32LE" in result.stdout
