from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_normalize_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_normalize",
            "fa_normalize.launch.py",
            "node_name:=fa_normalize",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )
def test_default_launch_config_keeps_normalize_as_dynamics_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_normalize"]["ros__parameters"]

    assert params["input_topic"] == "fa_normalize/input"
    assert params["output_topic"] == "fa_normalize/output"
    assert params["input_stream_id"] == "audio/noise_gated/mic"
    assert params["output"]["stream_id"] == "audio/normalized/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["normalize"]["target_peak_linear"] == 0.9
    assert params["normalize"]["silence_threshold_linear"] == 0.0001
    assert 0.0 < params["normalize"]["target_peak_linear"] <= 1.0
    assert 0.0 <= params["normalize"]["silence_threshold_linear"]
    assert params["normalize"]["silence_threshold_linear"] < params["normalize"]["target_peak_linear"]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert "resample" not in params
    assert "limiter" not in params
    assert "gain" not in params


def test_launch_fails_closed_when_target_peak_is_zero(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_normalize"]["ros__parameters"]["normalize"]["target_peak_linear"] = 0.0
    config_path = tmp_path / "zero_target_peak.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_normalize_launch(config_path)

    assert "process has died" in result.stdout
    assert "normalize.target_peak_linear must be finite and in (0.0, 1.0]" in result.stdout


def test_launch_fails_closed_when_silence_threshold_is_invalid(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_normalize"]["ros__parameters"]["normalize"]["silence_threshold_linear"] = 0.9
    config_path = tmp_path / "invalid_silence_threshold.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_normalize_launch(config_path)

    assert "process has died" in result.stdout
    assert (
        "normalize.silence_threshold_linear must be finite, >= 0.0, and < normalize.target_peak_linear"
        in result.stdout
    )


def test_launch_fails_closed_when_encoding_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_normalize"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_normalize_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_normalize requires expected.encoding=FLOAT32LE" in result.stdout


def test_launch_fails_closed_when_stream_ids_are_not_distinct(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_normalize"]["ros__parameters"]["output"]["stream_id"] = "audio/noise_gated/mic"
    config_path = tmp_path / "same_stream_ids.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_normalize_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id and output.stream_id must be distinct" in result.stdout


def test_launch_fails_closed_when_stream_id_matches_topic(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_normalize"]["ros__parameters"]["input_stream_id"] = "fa_normalize/output"
    config_path = tmp_path / "stream_id_matches_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_normalize_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id must be distinct from ROS topics" in result.stdout


def test_launch_fails_closed_when_output_stream_id_matches_topic(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_normalize"]["ros__parameters"]["output"]["stream_id"] = "fa_normalize/input"
    config_path = tmp_path / "output_stream_id_matches_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_normalize_launch(config_path)

    assert "process has died" in result.stdout
    assert "output.stream_id must be distinct from ROS topics" in result.stdout


def test_launch_fails_closed_when_stream_ids_match_after_slash_normalization(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_normalize"]["ros__parameters"]["output"]["stream_id"] = "/audio/noise_gated/mic"
    config_path = tmp_path / "slash_normalized_same_stream_ids.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_normalize_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id and output.stream_id must be distinct" in result.stdout
