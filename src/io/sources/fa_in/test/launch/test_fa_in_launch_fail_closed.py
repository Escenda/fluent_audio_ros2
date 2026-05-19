from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TypeAlias

import pytest
import yaml

YamlScalar: TypeAlias = str | int | bool | float | None
YamlValue: TypeAlias = YamlScalar | list["YamlValue"] | dict[str, "YamlValue"]
YamlConfig: TypeAlias = dict[str, YamlValue]


def _run_fa_in_launch(*launch_args: str) -> str:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    result = subprocess.run(
        [ros2, "launch", "fa_in", "fa_in.launch.py", *launch_args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )
    return result.stdout


def _write_config(path: Path, params: YamlConfig) -> Path:
    config: YamlConfig = {"fa_in": {"ros__parameters": params}}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_network_backend_config(
    path: Path,
    *,
    endpoint_uri: str,
    transport_identity: str,
    source_id: str,
    max_packet_bytes: int,
    polling_period_ms: int,
    source_timeout_ms: int = 100,
) -> Path:
    params: YamlConfig = {
        "backend.name": "network_pcm_receiver",
        "endpoint.uri": endpoint_uri,
        "transport.identity": transport_identity,
        "audio.source_id": source_id,
        "network.max_packet_bytes": max_packet_bytes,
        "polling.period_ms": polling_period_ms,
        "network.source_timeout_ms": source_timeout_ms,
        "output_topic": "fa_in/network_output",
        "audio.sample_rate": 16000,
        "audio.channels": 1,
        "audio.bit_depth": 16,
        "audio.chunk_ms": 20,
        "audio.encoding": "PCM16LE",
        "audio.stream_id": "audio/network/input",
        "audio.layout": "interleaved",
        "audio.qos.depth": 10,
        "audio.qos.reliable": False,
        "startup.required_subscribers": 0,
        "startup.subscriber_wait_timeout_ms": 0,
        "diagnostics.qos.depth": 10,
        "diagnostics.qos.reliable": False,
        "diagnostics.publish_period_ms": 1000,
    }
    return _write_config(path, params)


def test_default_launch_fails_closed_without_explicit_source_binding() -> None:
    default_config = Path(__file__).parents[2] / "config" / "default.yaml"
    output = _run_fa_in_launch(
        "node_name:=fa_in",
        f"config_file:={default_config}",
    )

    assert "process has died" in output
    assert "exit code 1" in output
    assert (
        "audio.device_selector.identifier is required when audio.device_selector.mode=id"
        in output
    )


def test_launch_fails_closed_when_params_file_is_missing(tmp_path: Path) -> None:
    missing_config = tmp_path / "missing.yaml"
    output = _run_fa_in_launch("node_name:=fa_in", f"config_file:={missing_config}")

    assert f"Parameter file path is not a file: {missing_config}" in output
    assert "process has died" in output
    assert "exit code 1" in output


def test_network_backend_launch_fails_closed_when_endpoint_uri_is_missing(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "missing_endpoint_uri.yaml",
        endpoint_uri="",
        transport_identity="audio/network/transport",
        source_id="audio/network/source",
        max_packet_bytes=320,
        polling_period_ms=5,
    )
    output = _run_fa_in_launch("node_name:=fa_in", f"config_file:={config_path}")

    assert "process has died" in output
    assert "endpoint.uri is required for backend.name=network_pcm_receiver" in output


def test_network_backend_launch_fails_closed_when_transport_identity_is_missing(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "missing_transport_identity.yaml",
        endpoint_uri="udp://127.0.0.1:40200",
        transport_identity="",
        source_id="audio/network/source",
        max_packet_bytes=320,
        polling_period_ms=5,
    )
    output = _run_fa_in_launch("node_name:=fa_in", f"config_file:={config_path}")

    assert "process has died" in output
    assert "transport.identity is required for backend.name=network_pcm_receiver" in output


def test_network_backend_launch_fails_closed_when_source_id_is_missing(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "missing_source_id.yaml",
        endpoint_uri="udp://127.0.0.1:40200",
        transport_identity="audio/network/transport",
        source_id="",
        max_packet_bytes=320,
        polling_period_ms=5,
    )
    output = _run_fa_in_launch("node_name:=fa_in", f"config_file:={config_path}")

    assert "process has died" in output
    assert "audio.source_id is required for backend.name=network_pcm_receiver" in output


def test_network_backend_launch_fails_closed_when_endpoint_uri_is_invalid(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "invalid_endpoint_uri.yaml",
        endpoint_uri="tcp://127.0.0.1:40200",
        transport_identity="audio/network/transport",
        source_id="audio/network/source",
        max_packet_bytes=320,
        polling_period_ms=5,
    )
    output = _run_fa_in_launch("node_name:=fa_in", f"config_file:={config_path}")

    assert "process has died" in output
    assert "endpoint.uri must use udp://host:port" in output


def test_network_backend_launch_fails_closed_when_max_packet_bytes_is_not_frame_aligned(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "unaligned_max_packet_bytes.yaml",
        endpoint_uri="udp://127.0.0.1:40200",
        transport_identity="audio/network/transport",
        source_id="audio/network/source",
        max_packet_bytes=3,
        polling_period_ms=5,
    )
    output = _run_fa_in_launch("node_name:=fa_in", f"config_file:={config_path}")

    assert "process has died" in output
    assert "network.max_packet_bytes must be divisible by expected frame byte size" in output


def test_network_backend_launch_fails_closed_when_polling_period_is_invalid(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "invalid_polling_period.yaml",
        endpoint_uri="udp://127.0.0.1:40200",
        transport_identity="audio/network/transport",
        source_id="audio/network/source",
        max_packet_bytes=320,
        polling_period_ms=0,
    )
    output = _run_fa_in_launch("node_name:=fa_in", f"config_file:={config_path}")

    assert "process has died" in output
    assert "polling.period_ms must be > 0" in output


def test_network_backend_launch_fails_closed_when_source_timeout_is_missing(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "missing_source_timeout.yaml",
        endpoint_uri="udp://127.0.0.1:40200",
        transport_identity="audio/network/transport",
        source_id="audio/network/source",
        max_packet_bytes=320,
        polling_period_ms=5,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = config["fa_in"]["ros__parameters"]
    del params["network.source_timeout_ms"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output = _run_fa_in_launch("node_name:=fa_in", f"config_file:={config_path}")

    assert "process has died" in output
    assert "network.source_timeout_ms is required" in output


def test_network_backend_launch_fails_closed_when_source_timeout_is_invalid(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "invalid_source_timeout.yaml",
        endpoint_uri="udp://127.0.0.1:40200",
        transport_identity="audio/network/transport",
        source_id="audio/network/source",
        max_packet_bytes=320,
        polling_period_ms=5,
        source_timeout_ms=0,
    )
    output = _run_fa_in_launch("node_name:=fa_in", f"config_file:={config_path}")

    assert "process has died" in output
    assert "network.source_timeout_ms must be > 0" in output


def test_network_backend_launch_fails_closed_when_polling_exceeds_source_timeout(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "polling_exceeds_source_timeout.yaml",
        endpoint_uri="udp://127.0.0.1:40200",
        transport_identity="audio/network/transport",
        source_id="audio/network/source",
        max_packet_bytes=320,
        polling_period_ms=20,
        source_timeout_ms=5,
    )
    output = _run_fa_in_launch("node_name:=fa_in", f"config_file:={config_path}")

    assert "process has died" in output
    assert "polling.period_ms must be <= network.source_timeout_ms" in output
