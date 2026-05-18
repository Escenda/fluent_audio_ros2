from pathlib import Path
import os
import shutil
import socket
import subprocess
import time
from typing import TypeAlias

import pytest
import yaml


YamlScalar: TypeAlias = str | int | float | bool | None
YamlMapping: TypeAlias = dict[str, "YamlValue"]
YamlSequence: TypeAlias = list["YamlValue"]
YamlValue: TypeAlias = YamlScalar | YamlMapping | YamlSequence


def _write_yaml(path: Path, value: YamlValue) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _allocate_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        _host, port = probe.getsockname()
        return int(port)


def _open_udp_receiver() -> tuple[socket.socket, int]:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.setblocking(False)
    _host, port = receiver.getsockname()
    return receiver, int(port)


def _write_network_io_params(
    tmp_path: Path,
    input_endpoint: str,
    output_endpoint: str,
) -> None:
    topic = "audio/e2e/network_pcm"
    _write_yaml(
        tmp_path / "fa_network_in.params.yaml",
        {
            "fa_network_in_e2e": {
                "ros__parameters": {
                    "backend.name": "network_pcm_receiver",
                    "endpoint.uri": input_endpoint,
                    "transport.identity": "network_in_e2e",
                    "output_topic": topic,
                    "audio.source_id": "network_e2e_source",
                    "audio.stream_id": topic,
                    "expected.sample_rate": 16000,
                    "expected.channels": 1,
                    "expected.encoding": "PCM16LE",
                    "expected.bit_depth": 16,
                    "expected.layout": "interleaved",
                    "network.max_packet_bytes": 1024,
                    "polling.period_ms": 10,
                    "qos.depth": 10,
                    "qos.reliable": True,
                    "diagnostics.publish_period_ms": 1000,
                }
            }
        },
    )
    _write_yaml(
        tmp_path / "fa_network_out.params.yaml",
        {
            "fa_network_out_e2e": {
                "ros__parameters": {
                    "backend.name": "network_pcm_sender",
                    "endpoint.uri": output_endpoint,
                    "transport.identity": "network_out_e2e",
                    "input_topic": topic,
                    "expected.sample_rate": 16000,
                    "expected.channels": 1,
                    "expected.encoding": "PCM16LE",
                    "expected.bit_depth": 16,
                    "expected.layout": "interleaved",
                    "qos.depth": 10,
                    "qos.reliable": True,
                    "diagnostics.publish_period_ms": 1000,
                }
            }
        },
    )


def _write_system_config(tmp_path: Path) -> Path:
    system_config = tmp_path / "network_io_system.yaml"
    _write_yaml(
        system_config,
        {
            "system": {
                "default_start_delay": 0.2,
                "inter_group_delay": 0.0,
            },
            "groups": [
                {
                    "id": "io",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_network_out",
                            "enable": True,
                            "package": "fa_network_out",
                            "exec": "fa_network_out_node",
                            "node_name": "fa_network_out_e2e",
                            "params_file": str(tmp_path / "fa_network_out.params.yaml"),
                        },
                        {
                            "id": "fa_network_in",
                            "enable": True,
                            "package": "fa_network_in",
                            "exec": "fa_network_in_node",
                            "node_name": "fa_network_in_e2e",
                            "params_file": str(tmp_path / "fa_network_in.params.yaml"),
                        },
                    ],
                }
            ],
        },
    )
    return system_config


def _wait_for_network_output(
    receiver: socket.socket,
    input_port: int,
    expected_payload: bytes,
) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            sender.sendto(expected_payload, ("127.0.0.1", input_port))
            try:
                packet, _address = receiver.recvfrom(1024)
            except BlockingIOError:
                time.sleep(0.05)
                continue
            if packet == expected_payload:
                return True
            return False
    return False


def _stop_process(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        process.terminate()
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _stderr = process.communicate(timeout=5)
    return stdout


def test_fluent_audio_system_launches_network_source_to_network_sink_e2e(
    tmp_path: Path,
) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for fluent_audio_system E2E launch")

    input_port = _allocate_udp_port()
    input_endpoint = f"udp://127.0.0.1:{input_port}"
    receiver, output_port = _open_udp_receiver()
    output_endpoint = f"udp://127.0.0.1:{output_port}"
    expected_payload = b"\x10\x00\x20\x00\x30\x00\x40\x00"

    try:
        _write_network_io_params(tmp_path, input_endpoint, output_endpoint)
        system_config = _write_system_config(tmp_path)

        process = subprocess.Popen(
            [
                ros2,
                "launch",
                "fluent_audio_system",
                "fluent_audio_system.launch.py",
                f"config:={system_config}",
                "fa_in_enabled:=false",
                "fa_out_enabled:=false",
            ],
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            matched = _wait_for_network_output(receiver, input_port, expected_payload)
            stdout = _stop_process(process)
            assert matched, stdout
        finally:
            if process.poll() is None:
                _stop_process(process)
    finally:
        receiver.close()
