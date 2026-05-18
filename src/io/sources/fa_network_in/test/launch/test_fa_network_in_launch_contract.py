from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_network_in_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_network_in",
            "fa_network_in.launch.py",
            "node_name:=fa_network_in",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_network_in.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'package="fa_network_in"' in launch_text
    assert 'executable="fa_network_in_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "endpoint.uri" not in launch_text
    assert "backend.name" not in launch_text
    assert "jitter" not in launch_text
    assert "codec" not in launch_text
    assert "drift" not in launch_text


def test_default_launch_config_declares_raw_pcm_network_source_contract() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_network_in"]["ros__parameters"]

    assert params["backend.name"] == "network_pcm_receiver"
    assert params["endpoint.uri"] == ""
    assert params["transport.identity"] == ""
    assert params["output_topic"] == "audio/network_in"
    assert params["audio"]["source_id"] == "network_source"
    assert params["audio"]["stream_id"] == "audio/network_in"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert params["expected"]["layout"] == "interleaved"
    assert params["network"]["max_packet_bytes"] == 3200
    assert params["polling"]["period_ms"] == 10


def test_launch_fails_closed_when_endpoint_uri_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_network_in"]["ros__parameters"]["transport.identity"] = "launch-test"
    config_path = tmp_path / "missing_endpoint_uri.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_network_in_launch(config_path)

    assert "process has died" in result.stdout
    assert "endpoint.uri is required" in result.stdout


def test_launch_fails_closed_when_transport_identity_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_network_in"]["ros__parameters"]["endpoint.uri"] = "udp://127.0.0.1:9"
    config_path = tmp_path / "missing_transport_identity.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_network_in_launch(config_path)

    assert "process has died" in result.stdout
    assert "transport.identity is required" in result.stdout
