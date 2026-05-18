from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_high_pass_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_high_pass",
            "fa_high_pass.launch.py",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_high_pass.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'package="fa_high_pass"' in launch_text
    assert 'executable="fa_high_pass_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "filter.cutoff_hz" not in launch_text
    assert "expected.sample_rate" not in launch_text
    assert "expected.channels" not in launch_text


def test_default_launch_config_keeps_high_pass_as_frequency_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_high_pass"]["ros__parameters"]

    assert params["input_topic"] == "audio/resample16k/mic"
    assert params["output_topic"] == "audio/high_pass/mic"
    assert params["filter"]["cutoff_hz"] == 80.0
    assert 0.0 < params["filter"]["cutoff_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert "resample" not in params
    assert "gain" not in params
    assert "normalize" not in params
    assert "limiter" not in params


def test_launch_fails_closed_when_cutoff_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_high_pass"]["ros__parameters"]["filter"]["cutoff_hz"] = 0.0
    config_path = tmp_path / "invalid_cutoff.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_high_pass_launch(config_path)

    assert "process has died" in result.stdout
    assert (
        "filter.cutoff_hz must be finite, > 0.0, and < expected.sample_rate / 2.0"
        in result.stdout
    )


def test_launch_fails_closed_when_encoding_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_high_pass"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_high_pass_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_high_pass requires expected.encoding=FLOAT32LE" in result.stdout
