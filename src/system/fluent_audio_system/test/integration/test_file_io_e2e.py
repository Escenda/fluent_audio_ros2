from pathlib import Path
import os
import shutil
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


def _write_file_io_params(tmp_path: Path, input_pcm: Path, output_pcm: Path) -> None:
    topic = "audio/e2e/file_pcm"
    stream_id = "audio/e2e/file_pcm_stream"
    _write_yaml(
        tmp_path / "fa_in.params.yaml",
        {
            "fa_in_e2e": {
                "ros__parameters": {
                    "backend.name": "pcm_file_reader",
                    "file.path": str(input_pcm),
                    "output_topic": topic,
                    "audio.source_id": "file_e2e_source",
                    "audio.sample_rate": 16000,
                    "audio.channels": 1,
                    "audio.encoding": "PCM16LE",
                    "audio.bit_depth": 16,
                    "audio.layout": "interleaved",
                    "audio.chunk_ms": 1,
                    "audio.stream_id": stream_id,
                    "playback.loop": False,
                    "audio.qos.depth": 10,
                    "audio.qos.reliable": True,
                    "diagnostics.publish_period_ms": 1000,
                    "diagnostics.qos.depth": 10,
                    "diagnostics.qos.reliable": True,
                }
            }
        },
    )
    _write_yaml(
        tmp_path / "fa_file_out.params.yaml",
        {
            "fa_file_out_e2e": {
                "ros__parameters": {
                    "backend.name": "pcm_file_writer",
                    "file.path": str(output_pcm),
                    "input_topic": topic,
                    "expected.sample_rate": 16000,
                    "expected.channels": 1,
                    "expected.encoding": "PCM16LE",
                    "expected.bit_depth": 16,
                    "expected.layout": "interleaved",
                    "overwrite.enabled": False,
                    "qos.depth": 10,
                    "qos.reliable": True,
                    "diagnostics.publish_period_ms": 1000,
                    "diagnostics.qos.depth": 10,
                    "diagnostics.qos.reliable": True,
                }
            }
        },
    )


def _write_system_config(tmp_path: Path) -> Path:
    system_config = tmp_path / "file_io_system.yaml"
    _write_yaml(
        system_config,
        {
            "system": {
                "default_start_delay": 1.0,
                "inter_group_delay": 0.0,
            },
            "groups": [
                {
                    "id": "io",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_file_out",
                            "enable": True,
                            "package": "fa_file_out",
                            "exec": "fa_file_out_node",
                            "node_name": "fa_file_out_e2e",
                            "params_file": str(tmp_path / "fa_file_out.params.yaml"),
                        },
                        {
                            "id": "fa_in",
                            "enable": True,
                            "package": "fa_in",
                            "exec": "fa_in_node",
                            "node_name": "fa_in_e2e",
                            "params_file": str(tmp_path / "fa_in.params.yaml"),
                        },
                    ],
                }
            ],
        },
    )
    return system_config


def _wait_for_output(output_pcm: Path, expected_payload: bytes) -> bool:
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        if output_pcm.exists() and output_pcm.read_bytes() == expected_payload:
            return True
        time.sleep(0.05)
    return output_pcm.exists() and output_pcm.read_bytes() == expected_payload


def _stop_process(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        process.terminate()
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _stderr = process.communicate(timeout=5)
    return stdout


def test_fluent_audio_system_launches_file_source_to_file_sink_e2e(tmp_path: Path) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for fluent_audio_system E2E launch")

    input_payload = b"\x10\x00\x20\x00\x30\x00\x40\x00"
    input_pcm = tmp_path / "input.pcm"
    output_pcm = tmp_path / "output.pcm"
    input_pcm.write_bytes(input_payload)
    _write_file_io_params(tmp_path, input_pcm, output_pcm)
    system_config = _write_system_config(tmp_path)

    process = subprocess.Popen(
        [
            ros2,
            "launch",
            "fluent_audio_system",
            "fluent_audio_system.launch.py",
            f"config:={system_config}",
            "fa_in_enabled:=true",
            "fa_out_enabled:=false",
            "fa_in_source_id:=disabled",
            "fa_out_sink_id:=disabled",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        matched = _wait_for_output(output_pcm, input_payload)
        stdout = _stop_process(process)
        assert matched, stdout
    finally:
        if process.poll() is None:
            _stop_process(process)
