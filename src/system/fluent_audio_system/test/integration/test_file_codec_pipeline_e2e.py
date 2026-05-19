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


def _write_codec_pipeline_params(tmp_path: Path, input_pcm: Path, output_pcm: Path) -> None:
    pcm16_input_topic = "audio/e2e/codec_pcm16_in"
    encoded_topic = "audio/e2e/codec_encoded"
    pcm16_decoded_topic = "audio/e2e/codec_pcm16_decoded"
    float32_topic = "audio/e2e/codec_float32"
    gained_topic = "audio/e2e/codec_gain_float32"
    pcm16_input_stream_id = "audio/e2e/codec_pcm16_in_stream"
    encoded_stream_id = "audio/e2e/codec_encoded_stream"
    pcm16_decoded_stream_id = "audio/e2e/codec_pcm16_decoded_stream"
    float32_stream_id = "audio/e2e/codec_float32_stream"
    gained_stream_id = "audio/e2e/codec_gain_float32_stream"

    _write_yaml(
        tmp_path / "fa_in.params.yaml",
        {
            "fa_in_codec_e2e": {
                "ros__parameters": {
                    "backend.name": "pcm_file_reader",
                    "file.path": str(input_pcm),
                    "output_topic": pcm16_input_topic,
                    "audio.source_id": "codec_e2e_source",
                    "audio.sample_rate": 16000,
                    "audio.channels": 1,
                    "audio.encoding": "PCM16LE",
                    "audio.bit_depth": 16,
                    "audio.layout": "interleaved",
                    "audio.chunk_ms": 1,
                    "audio.stream_id": pcm16_input_stream_id,
                    "playback.loop": False,
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
        tmp_path / "fa_encode.params.yaml",
        {
            "fa_encode_codec_e2e": {
                "ros__parameters": {
                    "backend.name": "external_codec_encoder",
                    "backend.command.executable": "/bin/cat",
                    "backend.command.arguments": ["--"],
                    "backend.command.timeout_ms": 3000,
                    "backend.command.max_output_bytes": 1048576,
                    "input_topic": pcm16_input_topic,
                    "output_topic": encoded_topic,
                    "input_stream_id": pcm16_input_stream_id,
                    "input.sample_rate": 16000,
                    "input.channels": 1,
                    "input.encoding": "PCM16LE",
                    "input.bit_depth": 16,
                    "input.layout": "interleaved",
                    "output.stream_id": encoded_stream_id,
                    "output.codec": "test-identity-codec",
                    "output.container": "raw",
                    "output.payload_format": "raw-bytes",
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
        tmp_path / "fa_decode.params.yaml",
        {
            "fa_decode_codec_e2e": {
                "ros__parameters": {
                    "backend.name": "external_codec_decoder",
                    "backend.command.executable": "/bin/cat",
                    "backend.command.arguments": ["--"],
                    "backend.command.timeout_ms": 3000,
                    "backend.command.max_output_bytes": 1048576,
                    "input_topic": encoded_topic,
                    "output_topic": pcm16_decoded_topic,
                    "input_stream_id": encoded_stream_id,
                    "input.codec": "test-identity-codec",
                    "input.container": "raw",
                    "input.payload_format": "raw-bytes",
                    "input.sample_rate": 16000,
                    "input.channels": 1,
                    "output.stream_id": pcm16_decoded_stream_id,
                    "output.sample_rate": 16000,
                    "output.channels": 1,
                    "output.encoding": "PCM16LE",
                    "output.bit_depth": 16,
                    "output.layout": "interleaved",
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
        tmp_path / "fa_sample_format.params.yaml",
        {
            "fa_sample_format_codec_e2e": {
                "ros__parameters": {
                    "input_topic": pcm16_decoded_topic,
                    "output_topic": float32_topic,
                    "input_stream_id": pcm16_decoded_stream_id,
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
            "fa_gain_codec_e2e": {
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
            "fa_out_codec_e2e": {
                "ros__parameters": {
                    "backend.name": "pcm_file_writer",
                    "file.path": str(output_pcm),
                    "input_topic": gained_topic,
                    "input_stream_id": gained_stream_id,
                    "playback_done_topic": "audio/e2e/codec_done",
                    "playback_control_service": "audio/e2e/codec_control",
                    "audio.sample_rate": 16000,
                    "audio.channels": 1,
                    "audio.encoding": "FLOAT32LE",
                    "audio.bit_depth": 32,
                    "audio.chunk_duration_ms": 1,
                    "queue.max_frames": 32,
                    "overwrite.enabled": False,
                    "audio.qos.depth": 10,
                    "audio.qos.reliable": True,
                    "lifecycle.qos.depth": 10,
                    "lifecycle.qos.reliable": True,
                }
            }
        },
    )


def _write_system_config(tmp_path: Path) -> Path:
    system_config = tmp_path / "file_codec_pipeline_system.yaml"
    _write_yaml(
        system_config,
        {
            "system": {
                "default_start_delay": 0.5,
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
                            "node_name": "fa_out_codec_e2e",
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
                            "node_name": "fa_sample_format_codec_e2e",
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
                            "node_name": "fa_gain_codec_e2e",
                            "params_file": str(tmp_path / "fa_gain.params.yaml"),
                        },
                    ],
                },
                {
                    "id": "format",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_decode",
                            "enable": True,
                            "package": "fa_decode",
                            "exec": "fa_decode_node",
                            "node_name": "fa_decode_codec_e2e",
                            "params_file": str(tmp_path / "fa_decode.params.yaml"),
                        },
                    ],
                },
                {
                    "id": "format",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_encode",
                            "enable": True,
                            "package": "fa_encode",
                            "exec": "fa_encode_node",
                            "node_name": "fa_encode_codec_e2e",
                            "params_file": str(tmp_path / "fa_encode.params.yaml"),
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
                            "node_name": "fa_in_codec_e2e",
                            "params_file": str(tmp_path / "fa_in.params.yaml"),
                        },
                    ],
                },
            ],
        },
    )
    return system_config


def _wait_for_output(output_pcm: Path, expected_payload: bytes) -> bool:
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        if output_pcm.exists() and output_pcm.read_bytes() == expected_payload:
            return True
        time.sleep(0.05)
    return output_pcm.exists() and output_pcm.read_bytes() == expected_payload


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


def test_fluent_audio_system_launches_explicit_codec_pipeline_e2e(
    tmp_path: Path,
) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for fluent_audio_system E2E launch")

    input_pcm = tmp_path / "input.pcm"
    output_pcm = tmp_path / "output.f32"
    input_payload = struct.pack("<hhhh", 8192, -8192, 4096, 0)
    expected_payload = struct.pack("<ffff", 0.5, -0.5, 0.25, 0.0)
    input_pcm.write_bytes(input_payload)
    _write_codec_pipeline_params(tmp_path, input_pcm, output_pcm)
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
        matched = _wait_for_output(output_pcm, expected_payload)
        stdout = _stop_process(process)
        assert matched, stdout
    finally:
        if process.poll() is None:
            _stop_process(process)
