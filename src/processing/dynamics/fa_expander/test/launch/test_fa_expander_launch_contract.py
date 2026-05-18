from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_expander_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_expander",
            "fa_expander.launch.py",
            "node_name:=fa_expander",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_expander.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert ("default_" + "value") not in launch_text
    assert ("FindPackage" + "Share") not in launch_text
    assert ("PathJoin" + "Substitution") not in launch_text
    assert 'package="fa_expander"' in launch_text
    assert 'executable="fa_expander_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "expander.threshold_linear" not in launch_text
    assert "expander.ratio" not in launch_text
    assert "expected.sample_rate" not in launch_text


def test_example_launch_config_keeps_expander_as_dynamics_node() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_expander"]["ros__parameters"]

    assert params["input_topic"] == "fa_expander/input"
    assert params["output_topic"] == "fa_expander/output"
    assert params["input_stream_id"] == "audio/noise_gated/mic"
    assert params["output"]["stream_id"] == "audio/expanded/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["expander"]["threshold_linear"] == 0.05
    assert params["expander"]["ratio"] == 2.0
    assert 0.0 < params["expander"]["threshold_linear"] < 1.0
    assert params["expander"]["ratio"] > 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert "resample" not in params
    assert "limiter" not in params
    assert "normalize" not in params
    assert "compressor" not in params


def test_launch_fails_closed_when_threshold_is_zero(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_expander"]["ros__parameters"]["expander"]["threshold_linear"] = 0.0
    config_path = tmp_path / "zero_threshold.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_expander_launch(config_path)

    assert "process has died" in result.stdout
    assert "expander.threshold_linear must be finite and in (0.0, 1.0)" in result.stdout


def test_launch_fails_closed_when_ratio_is_unity(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_expander"]["ros__parameters"]["expander"]["ratio"] = 1.0
    config_path = tmp_path / "unity_ratio.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_expander_launch(config_path)

    assert "process has died" in result.stdout
    assert "expander.ratio must be finite and > 1.0" in result.stdout


def test_launch_fails_closed_when_encoding_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_expander"]["ros__parameters"]["expected"]["encoding"] = "PCM16LE"
    config_path = tmp_path / "wrong_encoding.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_expander_launch(config_path)

    assert "process has died" in result.stdout
    assert "fa_expander requires expected.encoding=FLOAT32LE" in result.stdout


def test_launch_fails_closed_when_stream_ids_are_not_distinct(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_expander"]["ros__parameters"]["output"]["stream_id"] = "audio/noise_gated/mic"
    config_path = tmp_path / "same_stream_ids.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_expander_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id and output.stream_id must be distinct" in result.stdout


def test_launch_fails_closed_when_stream_id_matches_topic(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_expander"]["ros__parameters"]["input_stream_id"] = "fa_expander/output"
    config_path = tmp_path / "stream_id_matches_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_expander_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id must be distinct from ROS topics" in result.stdout


def test_launch_fails_closed_when_output_stream_id_matches_topic(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_expander"]["ros__parameters"]["output"]["stream_id"] = "fa_expander/input"
    config_path = tmp_path / "output_stream_id_matches_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_expander_launch(config_path)

    assert "process has died" in result.stdout
    assert "output.stream_id must be distinct from ROS topics" in result.stdout


def test_launch_fails_closed_when_stream_ids_match_after_slash_normalization(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_expander"]["ros__parameters"]["output"]["stream_id"] = "/audio/noise_gated/mic"
    config_path = tmp_path / "slash_normalized_same_stream_ids.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_expander_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id and output.stream_id must be distinct" in result.stdout
