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


def _run_fa_sample_format_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_sample_format",
            "fa_sample_format.launch.py",
            "node_name:=fa_sample_format",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )
def test_default_launch_config_keeps_sample_format_as_explicit_format_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_sample_format"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"
    assert params["output_topic"] == "audio/sample_format/mic"
    assert params["input_stream_id"] == "audio/raw/mic"
    assert params["output"]["stream_id"] == "audio/float32/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input"]["encoding"] == "PCM16LE"
    assert params["input"]["bit_depth"] == 16
    assert params["output"]["encoding"] == "FLOAT32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["layout"] == "interleaved"


def test_launch_fails_closed_when_conversion_is_unsupported(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_sample_format"]["ros__parameters"]["input"]["encoding"] = "MULAW"
    config_path = _write_launch_config(tmp_path, "unsupported_conversion.yaml", config)

    result = _run_fa_sample_format_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_sample_format supports only PCM16LE/16" in result.stdout


def test_launch_fails_closed_when_input_topic_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_sample_format"]["ros__parameters"]["input_topic"] = ""
    config_path = _write_launch_config(tmp_path, "missing_input_topic.yaml", config)

    result = _run_fa_sample_format_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_topic is required" in result.stdout


def test_launch_fails_closed_when_output_topic_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_sample_format"]["ros__parameters"]["output_topic"] = ""
    config_path = _write_launch_config(tmp_path, "missing_output_topic.yaml", config)

    result = _run_fa_sample_format_launch(config_path)

    assert "process has died" in result.stdout
    assert "output_topic is required" in result.stdout


def test_launch_fails_closed_when_expected_contract_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_sample_format"]["ros__parameters"]["expected"]["sample_rate"] = 0
    config_path = _write_launch_config(tmp_path, "invalid_expected_contract.yaml", config)

    result = _run_fa_sample_format_launch(config_path)

    assert "process has died" in result.stdout
    assert "expected.sample_rate must be > 0" in result.stdout


def test_launch_fails_closed_when_expected_channels_are_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_sample_format"]["ros__parameters"]["expected"]["channels"] = 0
    config_path = _write_launch_config(tmp_path, "invalid_expected_channels.yaml", config)

    result = _run_fa_sample_format_launch(config_path)

    assert "process has died" in result.stdout
    assert "expected.channels must be > 0" in result.stdout


def test_launch_fails_closed_when_expected_layout_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_sample_format"]["ros__parameters"]["expected"]["layout"] = "planar"
    config_path = _write_launch_config(tmp_path, "invalid_expected_layout.yaml", config)

    result = _run_fa_sample_format_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_sample_format requires expected.layout=interleaved" in result.stdout
