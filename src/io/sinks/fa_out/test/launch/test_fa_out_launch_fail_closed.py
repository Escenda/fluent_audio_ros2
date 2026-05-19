from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TypeAlias

import pytest
import yaml

YamlValue: TypeAlias = str | int | bool | float | None | list["YamlValue"] | dict[str, "YamlValue"]
YamlConfig: TypeAlias = dict[str, YamlValue]


def _run_fa_out_launch(*launch_args: str) -> str:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    result = subprocess.run(
        [ros2, "launch", "fa_out", "fa_out.launch.py", *launch_args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )
    return result.stdout


def _write_config(path: Path, params: YamlConfig) -> Path:
    config: YamlConfig = {"fa_out": {"ros__parameters": params}}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_file_backend_config(path: Path, target: str, *, overwrite_enabled: bool) -> Path:
    params: YamlConfig = {
        "backend.name": "pcm_file_writer",
        "file.path": target,
        "input_topic": "fa_out/file_input",
        "input_stream_id": "audio/file/output",
        "playback_done_topic": "fa_out/file_playback_done",
        "playback_control_service": "fa_out/file_playback_control",
        "audio.encoding": "PCM16LE",
        "audio.sample_rate": 16000,
        "audio.channels": 1,
        "audio.bit_depth": 16,
        "audio.chunk_duration_ms": 1,
        "queue.max_frames": 32,
        "audio.qos.depth": 10,
        "audio.qos.reliable": True,
        "lifecycle.qos.depth": 10,
        "lifecycle.qos.reliable": True,
        "overwrite.enabled": overwrite_enabled,
    }
    return _write_config(path, params)


def _write_network_backend_config(
    path: Path,
    *,
    endpoint_uri: str,
    transport_identity: str,
    max_packet_bytes: int,
) -> Path:
    params: YamlConfig = {
        "backend.name": "network_pcm_sender",
        "endpoint.uri": endpoint_uri,
        "transport.identity": transport_identity,
        "network.max_packet_bytes": max_packet_bytes,
        "input_topic": "fa_out/network_input",
        "input_stream_id": "audio/network/output",
        "playback_done_topic": "fa_out/network_playback_done",
        "playback_control_service": "fa_out/network_playback_control",
        "audio.encoding": "PCM16LE",
        "audio.sample_rate": 16000,
        "audio.channels": 1,
        "audio.bit_depth": 16,
        "audio.chunk_duration_ms": 1,
        "queue.max_frames": 32,
        "audio.qos.depth": 10,
        "audio.qos.reliable": True,
        "lifecycle.qos.depth": 10,
        "lifecycle.qos.reliable": True,
    }
    return _write_config(path, params)


def test_default_launch_fails_closed_without_explicit_sink_binding() -> None:
    default_config = Path(__file__).parents[2] / "config" / "default.yaml"
    output = _run_fa_out_launch(
        "node_name:=fa_out",
        f"config_file:={default_config}",
    )

    assert "process has died" in output
    assert "exit code 1" in output
    assert "audio.device_id is required for backend.name=alsa_playback" in output


def test_launch_fails_closed_when_params_file_is_missing(tmp_path: Path) -> None:
    missing_config = tmp_path / "missing.yaml"
    output = _run_fa_out_launch("node_name:=fa_out", f"config_file:={missing_config}")

    assert f"Parameter file path is not a file: {missing_config}" in output
    assert "process has died" in output
    assert "exit code 1" in output


def test_file_backend_launch_fails_closed_when_file_path_is_missing(tmp_path: Path) -> None:
    config_path = _write_file_backend_config(
        tmp_path / "missing_file_path.yaml",
        "",
        overwrite_enabled=False,
    )
    output = _run_fa_out_launch("node_name:=fa_out", f"config_file:={config_path}")

    assert "process has died" in output
    assert "file.path is required for backend.name=pcm_file_writer" in output


def test_file_backend_launch_fails_closed_when_overwrite_disabled_target_exists(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.pcm"
    target.write_bytes(b"\x00\x00")
    config_path = _write_file_backend_config(
        tmp_path / "target_exists.yaml",
        str(target),
        overwrite_enabled=False,
    )
    output = _run_fa_out_launch("node_name:=fa_out", f"config_file:={config_path}")

    assert "process has died" in output
    assert "overwrite.enabled=false" in output


def test_network_backend_launch_fails_closed_when_endpoint_uri_is_missing(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "missing_endpoint_uri.yaml",
        endpoint_uri="",
        transport_identity="audio/network/transport",
        max_packet_bytes=320,
    )
    output = _run_fa_out_launch("node_name:=fa_out", f"config_file:={config_path}")

    assert "process has died" in output
    assert "endpoint.uri is required for backend.name=network_pcm_sender" in output


def test_network_backend_launch_fails_closed_when_transport_identity_is_missing(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "missing_transport_identity.yaml",
        endpoint_uri="udp://127.0.0.1:40100",
        transport_identity="",
        max_packet_bytes=320,
    )
    output = _run_fa_out_launch("node_name:=fa_out", f"config_file:={config_path}")

    assert "process has died" in output
    assert "transport.identity is required for backend.name=network_pcm_sender" in output


def test_network_backend_launch_fails_closed_when_endpoint_uri_is_invalid(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "invalid_endpoint_uri.yaml",
        endpoint_uri="tcp://127.0.0.1:40100",
        transport_identity="audio/network/transport",
        max_packet_bytes=320,
    )
    output = _run_fa_out_launch("node_name:=fa_out", f"config_file:={config_path}")

    assert "process has died" in output
    assert "endpoint.uri must use udp://host:port" in output


def test_network_backend_launch_fails_closed_when_max_packet_bytes_is_not_frame_aligned(
    tmp_path: Path,
) -> None:
    config_path = _write_network_backend_config(
        tmp_path / "unaligned_max_packet_bytes.yaml",
        endpoint_uri="udp://127.0.0.1:40100",
        transport_identity="audio/network/transport",
        max_packet_bytes=3,
    )
    output = _run_fa_out_launch("node_name:=fa_out", f"config_file:={config_path}")

    assert "process has died" in output
    assert "network.max_packet_bytes must be divisible by expected frame byte size" in output
