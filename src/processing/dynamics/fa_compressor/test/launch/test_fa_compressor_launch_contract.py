from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_compressor_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_compressor",
            "fa_compressor.launch.py",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_compressor.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'default_value="fa_compressor"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'FindPackageShare("fa_compressor"), "config", "default.yaml"' in launch_text
    assert 'package="fa_compressor"' in launch_text
    assert 'executable="fa_compressor_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "compressor.threshold_linear" not in launch_text
    assert "compressor.ratio" not in launch_text
    assert "compressor.makeup_gain_linear" not in launch_text
    assert "expected.sample_rate" not in launch_text


def test_default_launch_config_keeps_compressor_as_dynamics_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_compressor"]["ros__parameters"]

    assert params["input_topic"] == "audio/normalized/mic"
    assert params["output_topic"] == "audio/compressed/mic"
    assert params["compressor"]["threshold_linear"] == 0.5
    assert params["compressor"]["ratio"] == 4.0
    assert params["compressor"]["makeup_gain_linear"] == 1.0
    assert 0.0 < params["compressor"]["threshold_linear"] < 1.0
    assert params["compressor"]["ratio"] > 1.0
    assert 0.0 < params["compressor"]["makeup_gain_linear"] <= 4.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert "resample" not in params
    assert "limiter" not in params
    assert "normalize" not in params


def test_launch_fails_closed_when_threshold_is_zero(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_compressor"]["ros__parameters"]["compressor"]["threshold_linear"] = 0.0
    config_path = tmp_path / "zero_threshold.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_compressor_launch(config_path)

    assert "process has died" in result.stdout
    assert "compressor.threshold_linear must be finite and in (0.0, 1.0)" in result.stdout


def test_launch_fails_closed_when_ratio_is_unity(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_compressor"]["ros__parameters"]["compressor"]["ratio"] = 1.0
    config_path = tmp_path / "unity_ratio.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_compressor_launch(config_path)

    assert "process has died" in result.stdout
    assert "compressor.ratio must be finite and > 1.0" in result.stdout


def test_launch_fails_closed_when_makeup_gain_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_compressor"]["ros__parameters"]["compressor"]["makeup_gain_linear"] = 4.1
    config_path = tmp_path / "invalid_makeup_gain.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_compressor_launch(config_path)

    assert "process has died" in result.stdout
    assert "compressor.makeup_gain_linear must be finite and in (0.0, 4.0]" in result.stdout


def test_launch_fails_closed_when_encoding_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_compressor"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_compressor_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_compressor requires expected.encoding=FLOAT32LE" in result.stdout
