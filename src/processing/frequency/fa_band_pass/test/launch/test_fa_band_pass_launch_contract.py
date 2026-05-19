from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_band_pass_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_band_pass",
            "fa_band_pass.launch.py",
            "node_name:=fa_band_pass",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )
def test_default_launch_config_keeps_band_pass_as_frequency_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_band_pass"]["ros__parameters"]

    assert params["input_topic"] == "fa_band_pass/input"
    assert params["output_topic"] == "fa_band_pass/output"
    assert params["input_stream_id"] == "audio/sample_format/mic"
    assert params["output"]["stream_id"] == "audio/band_pass/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["filter"]["low_cut_hz"] == 80.0
    assert params["filter"]["high_cut_hz"] == 3400.0
    assert 0.0 < params["filter"]["low_cut_hz"] < params["filter"]["high_cut_hz"]
    assert params["filter"]["high_cut_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert "resample" not in params
    assert "gain" not in params
    assert "normalize" not in params
    assert "limiter" not in params


def test_launch_fails_closed_when_low_cut_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_band_pass"]["ros__parameters"]["filter"]["low_cut_hz"] = 0.0
    config_path = tmp_path / "invalid_low_cut.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_band_pass_launch(config_path)

    assert "process has died" in result.stdout
    assert "filter.low_cut_hz must be finite and > 0.0" in result.stdout


def test_launch_fails_closed_when_high_cut_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_band_pass"]["ros__parameters"]["filter"]["high_cut_hz"] = 80.0
    config_path = tmp_path / "invalid_high_cut.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_band_pass_launch(config_path)

    assert "process has died" in result.stdout
    assert (
        "filter.high_cut_hz must be finite, > filter.low_cut_hz, and < expected.sample_rate / 2.0"
        in result.stdout
    )


def test_launch_fails_closed_when_encoding_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_band_pass"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_band_pass_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_band_pass requires expected.encoding=FLOAT32LE" in result.stdout
