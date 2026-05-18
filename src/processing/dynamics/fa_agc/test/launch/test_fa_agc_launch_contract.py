from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_agc_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_agc",
            "fa_agc.launch.py",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_agc.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'default_value="fa_agc"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'FindPackageShare("fa_agc"), "config", "default.yaml"' in launch_text
    assert 'package="fa_agc"' in launch_text
    assert 'executable="fa_agc_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "agc.target_rms" not in launch_text
    assert "agc.min_gain" not in launch_text
    assert "agc.max_gain" not in launch_text
    assert "expected.sample_rate" not in launch_text


def test_default_launch_config_keeps_agc_as_dynamics_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_agc"]["ros__parameters"]

    assert params["input_topic"] == "audio/compressed/mic"
    assert params["output_topic"] == "audio/agc/mic"
    assert params["agc"]["target_rms"] == 0.1
    assert params["agc"]["min_gain"] == 0.25
    assert params["agc"]["max_gain"] == 4.0
    assert params["agc"]["attack_ms"] == 10.0
    assert params["agc"]["release_ms"] == 250.0
    assert 0.0 < params["agc"]["target_rms"] <= 1.0
    assert 0.0 < params["agc"]["min_gain"] <= 1.0
    assert params["agc"]["max_gain"] >= 1.0
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert "resample" not in params
    assert "limiter" not in params
    assert "normalize" not in params
    assert "compressor" not in params


def test_launch_fails_closed_when_target_rms_is_zero(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_agc"]["ros__parameters"]["agc"]["target_rms"] = 0.0
    config_path = tmp_path / "zero_target_rms.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_agc_launch(config_path)

    assert "process has died" in result.stdout
    assert "agc.target_rms must be finite and in (0.0, 1.0]" in result.stdout


def test_launch_fails_closed_when_initial_gain_range_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_agc"]["ros__parameters"]["agc"]["min_gain"] = 1.1
    config["fa_agc"]["ros__parameters"]["agc"]["max_gain"] = 4.0
    config_path = tmp_path / "invalid_initial_gain.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_agc_launch(config_path)

    assert "process has died" in result.stdout
    assert "agc.min_gain <= 1.0 <= agc.max_gain is required for initial gain" in result.stdout


def test_launch_fails_closed_when_attack_is_zero(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_agc"]["ros__parameters"]["agc"]["attack_ms"] = 0.0
    config_path = tmp_path / "zero_attack.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_agc_launch(config_path)

    assert "process has died" in result.stdout
    assert "agc.attack_ms must be finite and > 0.0" in result.stdout


def test_launch_fails_closed_when_encoding_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_agc"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_agc_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_agc requires expected.encoding=FLOAT32LE" in result.stdout
