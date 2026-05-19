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


def _write_yaml(path: Path, value: YamlValue) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _stop_process(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        process.terminate()
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _stderr = process.communicate(timeout=5)
    return stdout


def _wait_for_service(node, client, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if client.wait_for_service(timeout_sec=0.1):
            return True
        time.sleep(0.02)
    return False


def _wait_for_subscription(publisher, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if publisher.get_subscription_count() > 0:
            return True
        time.sleep(0.02)
    return False


def _call_record_service(node, client, command: str, file_path: Path):
    from fa_interfaces.srv import Record

    request = Record.Request()
    request.command = command
    request.file_path = str(file_path)
    future = client.call_async(request)

    import rclpy

    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    return future.result()


def _recording_contains_payload(output_wav: Path, payload: bytes) -> bool:
    if not output_wav.exists():
        return False
    recorded = output_wav.read_bytes()
    return len(recorded) > 44 and payload in recorded


def _uint16_le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], byteorder="little", signed=False)


def _uint32_le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], byteorder="little", signed=False)


def _assert_pcm16_wav(recorded: bytes, payload: bytes) -> None:
    assert recorded[:4] == b"RIFF"
    assert recorded[8:12] == b"WAVE"
    assert recorded[12:16] == b"fmt "
    assert _uint32_le(recorded, 16) == 16
    assert _uint16_le(recorded, 20) == 1
    assert _uint16_le(recorded, 22) == 1
    assert _uint32_le(recorded, 24) == 16000
    assert _uint16_le(recorded, 34) == 16
    assert recorded[36:40] == b"data"
    assert _uint32_le(recorded, 40) == len(payload)
    assert recorded[44:] == payload


def _audio_frame(payload: bytes):
    from fa_interfaces.msg import AudioFrame

    msg = AudioFrame()
    msg.source_id = "fa_record_integration_source"
    msg.stream_id = "fa_record_integration_stream"
    msg.encoding = "PCM16LE"
    msg.sample_rate = 16000
    msg.channels = 1
    msg.bit_depth = 16
    msg.layout = "interleaved"
    msg.data = list(payload)
    msg.epoch = 1
    return msg


def test_fa_record_records_audio_frame_payload_to_wav_file(tmp_path: Path) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for fa_record graph integration")

    import rclpy
    from fa_interfaces.msg import AudioFrame
    from fa_interfaces.srv import Record

    input_topic = "audio/integration/fa_record_input"
    output_wav = tmp_path / "recorded.wav"
    config_path = _write_yaml(
        tmp_path / "fa_record.params.yaml",
        {
            "fa_record_integration": {
                "ros__parameters": {
                    "input_topic": input_topic,
                    "input.qos.depth": 10,
                    "input.qos.reliable": True,
                }
            }
        },
    )

    process = subprocess.Popen(
        [
            ros2,
            "launch",
            "fa_record",
            "fa_record.launch.py",
            "node_name:=fa_record_integration",
            f"config_file:={config_path}",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    rclpy.init()
    node = rclpy.create_node("fa_record_graph_test")
    client = node.create_client(Record, "/record")
    publisher = node.create_publisher(AudioFrame, input_topic, 10)
    payload = b"\x01\x00\x02\x00\x03\x00\x04\x00"

    try:
        assert _wait_for_service(node, client, 8.0), _stop_process(process)
        assert _wait_for_subscription(publisher, 5.0), _stop_process(process)

        start_response = _call_record_service(node, client, "start", output_wav)
        assert start_response is not None
        assert start_response.success, start_response.message

        publisher.publish(_audio_frame(payload))
        deadline = time.monotonic() + 5.0
        while (
            time.monotonic() < deadline
            and not _recording_contains_payload(output_wav, payload)
        ):
            rclpy.spin_once(node, timeout_sec=0.05)

        stop_response = _call_record_service(node, client, "stop", output_wav)
        assert stop_response is not None
        assert stop_response.success, stop_response.message

        _assert_pcm16_wav(output_wav.read_bytes(), payload)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if process.poll() is None:
            _stop_process(process)
