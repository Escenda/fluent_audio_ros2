from pathlib import Path
import os
import signal
import shutil
import struct
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


def _write_pipeline_params(tmp_path: Path, input_pcm: Path, output_pcm: Path) -> None:
    pcm16_topic = "audio/e2e/pipeline_pcm16"
    float32_topic = "audio/e2e/pipeline_float32"
    gained_topic = "audio/e2e/pipeline_gain_float32"
    pcm16_stream_id = "audio/e2e/pipeline_pcm16_stream"
    float32_stream_id = "audio/e2e/pipeline_float32_stream"
    gained_stream_id = "audio/e2e/pipeline_gain_float32_stream"

    _write_yaml(
        tmp_path / "fa_in.params.yaml",
        {
            "fa_in_pipeline_e2e": {
                "ros__parameters": {
                    "backend.name": "pcm_file_reader",
                    "audio.device_selector.mode": "",
                    "audio.device_selector.identifier": "",
                    "audio.device_selector.index": -1,
                    "file.path": str(input_pcm),
                    "endpoint.uri": "",
                    "transport.identity": "",
                    "output_topic": pcm16_topic,
                    "audio.source_id": "pipeline_e2e_source",
                    "audio.sample_rate": 16000,
                    "audio.channels": 1,
                    "audio.encoding": "PCM16LE",
                    "audio.bit_depth": 16,
                    "audio.layout": "interleaved",
                    "audio.chunk_ms": 1,
                    "audio.stream_id": pcm16_stream_id,
                    "playback.loop": True,
                    "network.max_packet_bytes": 0,
                    "polling.period_ms": 0,
                    "network.source_timeout_ms": 0,
                    "audio.qos.depth": 10,
                    "audio.qos.reliable": True,
                    "startup.required_subscribers": 1,
                    "startup.subscriber_wait_timeout_ms": 5000,
                    "diagnostics.publish_period_ms": 1000,
                    "diagnostics.qos.depth": 10,
                    "diagnostics.qos.reliable": True,
                }
            }
        },
    )
    _write_yaml(
        tmp_path / "fa_sample_format.params.yaml",
        {
            "fa_sample_format_pipeline_e2e": {
                "ros__parameters": {
                    "input_topic": pcm16_topic,
                    "output_topic": float32_topic,
                    "input_stream_id": pcm16_stream_id,
                    "input.encoding": "PCM16LE",
                    "input.bit_depth": 16,
                    "output.stream_id": float32_stream_id,
                    "output.encoding": "FLOAT32LE",
                    "output.bit_depth": 32,
                    "expected.sample_rate": 16000,
                    "expected.channels": 1,
                    "expected.layout": "interleaved",
                    "qos.depth": 10,
                    "qos.reliable": True,
                    "diagnostics.publish_period_ms": 1000,
                    "diagnostics.qos.depth": 10,
                    "diagnostics.qos.reliable": True,
                }
            }
        },
    )
    _write_yaml(
        tmp_path / "fa_gain.params.yaml",
        {
            "fa_gain_pipeline_e2e": {
                "ros__parameters": {
                    "input_topic": float32_topic,
                    "output_topic": gained_topic,
                    "input_stream_id": float32_stream_id,
                    "output.stream_id": gained_stream_id,
                    "gain.linear": 2.0,
                    "expected.sample_rate": 16000,
                    "expected.channels": 1,
                    "expected.encoding": "FLOAT32LE",
                    "expected.bit_depth": 32,
                    "expected.layout": "interleaved",
                    "qos.depth": 10,
                    "qos.reliable": True,
                    "diagnostics.publish_period_ms": 1000,
                    "diagnostics.qos.depth": 10,
                    "diagnostics.qos.reliable": True,
                }
            }
        },
    )
    _write_yaml(
        tmp_path / "fa_out.params.yaml",
        {
            "fa_out_pipeline_e2e": {
                "ros__parameters": {
                    "backend.name": "pcm_file_writer",
                    "audio.device_id": "",
                    "file.path": str(output_pcm),
                    "endpoint.uri": "",
                    "transport.identity": "",
                    "input_topic": gained_topic,
                    "input_stream_id": gained_stream_id,
                    "playback_done_topic": "audio/e2e/pipeline_done",
                    "playback_control_service": "audio/e2e/pipeline_control",
                    "audio.sample_rate": 16000,
                    "audio.channels": 1,
                    "audio.encoding": "FLOAT32LE",
                    "audio.bit_depth": 32,
                    "audio.chunk_duration_ms": 1,
                    "queue.max_frames": 32,
                    "overwrite.enabled": False,
                    "network.max_packet_bytes": 0,
                    "audio.alsa.buffer_frames": 0,
                    "audio.alsa.period_frames": 0,
                    "audio.qos.depth": 10,
                    "audio.qos.reliable": True,
                    "lifecycle.qos.depth": 10,
                    "lifecycle.qos.reliable": True,
                }
            }
        },
    )


def _write_system_config(tmp_path: Path) -> Path:
    system_config = tmp_path / "file_processing_pipeline_system.yaml"
    _write_yaml(
        system_config,
        {
            "system": {
                "default_start_delay": 3.0,
                "inter_group_delay": 0.0,
            },
            "groups": [
                {
                    "id": "io",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_out",
                            "enable": True,
                            "package": "fa_out",
                            "exec": "fa_out_node",
                            "node_name": "fa_out_pipeline_e2e",
                            "params_file": str(tmp_path / "fa_out.params.yaml"),
                        },
                    ],
                },
                {
                    "id": "format",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_sample_format",
                            "enable": True,
                            "package": "fa_sample_format",
                            "exec": "fa_sample_format_node",
                            "node_name": "fa_sample_format_pipeline_e2e",
                            "params_file": str(tmp_path / "fa_sample_format.params.yaml"),
                        },
                    ],
                },
                {
                    "id": "dynamics",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_gain",
                            "enable": True,
                            "package": "fa_gain",
                            "exec": "fa_gain_node",
                            "node_name": "fa_gain_pipeline_e2e",
                            "params_file": str(tmp_path / "fa_gain.params.yaml"),
                        },
                    ],
                },
                {
                    "id": "io",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_in",
                            "enable": True,
                            "package": "fa_in",
                            "exec": "fa_in_node",
                            "node_name": "fa_in_pipeline_e2e",
                            "params_file": str(tmp_path / "fa_in.params.yaml"),
                        },
                    ],
                },
            ],
        },
    )
    return system_config


def _matches_looped_payload(payload: bytes, expected_payload: bytes) -> bool:
    if payload == expected_payload:
        return True
    if not payload or len(payload) % len(expected_payload) != 0:
        return False
    return all(
        payload[offset : offset + len(expected_payload)] == expected_payload
        for offset in range(0, len(payload), len(expected_payload))
    )


def _wait_for_output(output_pcm: Path, expected_payload: bytes) -> bool:
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        if output_pcm.exists():
            if _matches_looped_payload(output_pcm.read_bytes(), expected_payload):
                return True
        time.sleep(0.05)
    return output_pcm.exists() and _matches_looped_payload(
        output_pcm.read_bytes(),
        expected_payload,
    )


def _stop_process(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _stderr = process.communicate(timeout=5)
    return stdout


def test_fluent_audio_system_launches_explicit_processing_pipeline_e2e(
    tmp_path: Path,
) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for fluent_audio_system E2E launch")

    input_pcm = tmp_path / "input.pcm"
    output_pcm = tmp_path / "output.f32"
    input_pcm.write_bytes(struct.pack("<hhhh", 8192, -8192, 4096, 0))
    expected_output = struct.pack("<ffff", 0.5, -0.5, 0.25, 0.0)
    _write_pipeline_params(tmp_path, input_pcm, output_pcm)
    system_config = _write_system_config(tmp_path)

    process = subprocess.Popen(
        [
            ros2,
            "launch",
            "fluent_audio_system",
            "fluent_audio_system.launch.py",
            f"config:={system_config}",
            "fa_in_enabled:=true",
            "fa_out_enabled:=true",
            "fa_in_source_id:=disabled",
            "fa_out_sink_id:=disabled",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    try:
        matched = _wait_for_output(output_pcm, expected_output)
        stdout = _stop_process(process)
        assert matched, stdout
    finally:
        if process.poll() is None:
            _stop_process(process)
