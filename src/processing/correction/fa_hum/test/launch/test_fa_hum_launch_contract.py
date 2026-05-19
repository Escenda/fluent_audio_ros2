from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_hum_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_hum",
            "fa_hum.launch.py",
            "node_name:=fa_hum",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )
def test_launch_fails_closed_when_input_topic_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_hum"]["ros__parameters"]["input_topic"] = ""
    config_path = tmp_path / "missing_input_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_hum_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_topic is required" in result.stdout


def test_launch_fails_closed_when_input_stream_id_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_hum"]["ros__parameters"]["input_stream_id"] = ""
    config_path = tmp_path / "missing_input_stream_id.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_hum_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id is required" in result.stdout


def test_launch_fails_closed_when_resolved_input_and_output_topics_match(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_hum"]["ros__parameters"]
    params["input_topic"] = "audio/hum_same"
    params["output_topic"] = "/audio/hum_same"
    config_path = tmp_path / "same_resolved_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_hum_launch(config_path)

    assert "process has died" in result.stdout
    assert "resolved input_topic and output_topic must be distinct" in result.stdout


def test_launch_fails_closed_when_stream_identity_matches_topic(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_hum"]["ros__parameters"]
    params["input_stream_id"] = params["input_topic"]
    config_path = tmp_path / "stream_matches_topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_hum_launch(config_path)

    assert "process has died" in result.stdout
    assert "input_stream_id must be distinct from ROS topics" in result.stdout


def test_launch_fails_closed_when_frequency_contract_is_wrong(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_hum"]["ros__parameters"]["hum"]["frequency_hz"] = 9000.0
    config_path = tmp_path / "bad_frequency.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_hum_launch(config_path)

    assert "process has died" in result.stdout
    assert "hum.frequency_hz must produce at least one harmonic below Nyquist" in result.stdout
