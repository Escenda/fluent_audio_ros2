from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import TypeAlias

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]
YamlValue: TypeAlias = str | int | bool | float | None | list["YamlValue"] | dict[str, "YamlValue"]
YamlConfig: TypeAlias = dict[str, YamlValue]


def _write_launch_config(tmp_path: Path, filename: str, config: YamlConfig) -> Path:
    config_path = tmp_path / filename
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _run_fa_resample_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_resample",
            "fa_resample.launch.py",
            "node_name:=fa_resample",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_resample.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'package="fa_resample"' in launch_text
    assert 'executable="fa_resample_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "target_sample_rate" not in launch_text
    assert "backend.name" not in launch_text


def test_default_launch_config_keeps_resample_as_explicit_format_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_resample"]["ros__parameters"]

    assert params["target_sample_rate"] == 16000
    assert params["input"]["encoding"] == "FLOAT32LE"
    assert params["input"]["bit_depth"] == 32
    assert params["input"]["layout"] == "interleaved"
    assert params["output"]["encoding"] == "FLOAT32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["mic"]["enabled"] is True
    assert params["mic"]["input_topic"] == "audio/frame"
    assert params["mic"]["output_topic"] == "audio/resample16k/mic"
    assert params["mic"]["input_stream_id"] == "audio/float32/mic"
    assert params["mic"]["output"]["stream_id"] == "audio/preprocessed/mono16k"
    assert params["mic"]["input_topic"] != params["mic"]["input_stream_id"]
    assert params["mic"]["output_topic"] != params["mic"]["output"]["stream_id"]


def test_launch_fails_closed_when_target_sample_rate_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_resample"]["ros__parameters"]["target_sample_rate"] = 0
    config_path = _write_launch_config(tmp_path, "invalid_target_sample_rate.yaml", config)

    result = _run_fa_resample_launch(config_path)

    assert "process has died" in result.stdout
    assert "target_sample_rate must be > 0" in result.stdout


def test_launch_fails_closed_when_enabled_mic_topic_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_resample"]["ros__parameters"]["mic"]["output_topic"] = ""
    config_path = _write_launch_config(tmp_path, "missing_mic_output_topic.yaml", config)

    result = _run_fa_resample_launch(config_path)

    assert "process has died" in result.stdout
    assert "mic.output_topic is required when mic.enabled=true" in result.stdout


@pytest.mark.parametrize(
    ("parameter_path", "value", "expected_error"),
    (
        (("input", "encoding"), "PCM16LE", "fa_resample input.encoding must be FLOAT32LE"),
        (("input", "bit_depth"), 16, "fa_resample input.bit_depth must be 32"),
        (("input", "layout"), "planar", "fa_resample input.layout must be interleaved"),
        (("output", "encoding"), "PCM16LE", "fa_resample output.encoding must be FLOAT32LE"),
        (("output", "bit_depth"), 16, "fa_resample output.bit_depth must be 32"),
    ),
)
def test_launch_fails_closed_when_format_contract_is_not_float32le(
    tmp_path: Path,
    parameter_path: tuple[str, str],
    value: str | int,
    expected_error: str,
) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    section, key = parameter_path
    config["fa_resample"]["ros__parameters"][section][key] = value
    config_path = _write_launch_config(tmp_path, "invalid_format_contract.yaml", config)

    result = _run_fa_resample_launch(config_path)

    assert "process has died" in result.stdout
    assert expected_error in result.stdout
