from pathlib import Path
import os
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


def _stop_processes(processes: list[subprocess.Popen[str]]) -> str:
    outputs: list[str] = []
    for process in processes:
        outputs.append(_stop_process(process))
    return "\n".join(outputs)


def _wait_for_subscription(publisher, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if publisher.get_subscription_count() > 0:
            return True
        time.sleep(0.02)
    return False


def _wait_for_publisher(node, topic: str, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if node.count_publishers(topic) > 0:
            return True
        time.sleep(0.02)
    return False


def _float32_payload(sample_count: int) -> bytes:
    payload = bytearray()
    for index in range(sample_count):
        sample = (index % 96) / 96.0
        payload.extend(struct.pack("<f", sample))
    return bytes(payload)


def _audio_frame(payload: bytes):
    from fa_interfaces.msg import AudioFrame

    msg = AudioFrame()
    msg.source_id = "resample_frame_buffer_source"
    msg.stream_id = "stream/resample_frame_buffer/float32_48k"
    msg.encoding = "FLOAT32LE"
    msg.sample_rate = 48000
    msg.channels = 1
    msg.bit_depth = 32
    msg.layout = "interleaved"
    msg.data = list(payload)
    msg.epoch = 9
    return msg


def test_resample_to_frame_buffer_launch_graph_publishes_fixed_chunk(
    tmp_path: Path,
) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch graph integration")

    import rclpy
    from fa_interfaces.msg import AudioFrame

    resample_input_topic = "/audio/integration/resample_frame_buffer/float32_48k"
    resample_output_topic = "/audio/integration/resample_frame_buffer/float32_16k"
    chunk_topic = "/audio/integration/resample_frame_buffer/chunk"

    resample_config = _write_yaml(
        tmp_path / "fa_resample.params.yaml",
        {
            "fa_resample_frame_buffer_integration": {
                "ros__parameters": {
                    "target_sample_rate": 16000,
                    "input": {
                        "encoding": "FLOAT32LE",
                        "bit_depth": 32,
                        "layout": "interleaved",
                    },
                    "output": {
                        "encoding": "FLOAT32LE",
                        "bit_depth": 32,
                    },
                    "qos": {
                        "depth": 10,
                        "reliable": True,
                    },
                    "diagnostics": {
                        "publish_period_ms": 1000,
                        "qos": {
                            "depth": 10,
                            "reliable": True,
                        },
                    },
                    "mic": {
                        "enabled": True,
                        "input_topic": resample_input_topic,
                        "output_topic": resample_output_topic,
                        "input_stream_id": "stream/resample_frame_buffer/float32_48k",
                        "output": {
                            "stream_id": "stream/resample_frame_buffer/float32_16k",
                        },
                    },
                    "ref": {
                        "enabled": False,
                        "input_topic": "/audio/integration/resample_frame_buffer/ref_in",
                        "output_topic": "/audio/integration/resample_frame_buffer/ref_out",
                        "input_stream_id": "stream/resample_frame_buffer/ref_float32_48k",
                        "output": {
                            "stream_id": "stream/resample_frame_buffer/ref_float32_16k",
                        },
                    },
                }
            }
        },
    )
    frame_buffer_config = _write_yaml(
        tmp_path / "fa_frame_buffer.params.yaml",
        {
            "fa_frame_buffer_integration": {
                "ros__parameters": {
                    "input_topic": resample_output_topic,
                    "output_topic": chunk_topic,
                    "input_stream_id": "stream/resample_frame_buffer/float32_16k",
                    "output": {
                        "stream_id": "stream/resample_frame_buffer/chunk_16k",
                    },
                    "expected": {
                        "sample_rate": 16000,
                        "channels": 1,
                        "encoding": "FLOAT32LE",
                        "bit_depth": 32,
                        "layout": "interleaved",
                    },
                    "buffering": {
                        "frames_per_chunk": 160,
                        "max_buffered_chunks": 2,
                    },
                    "qos": {
                        "depth": 10,
                        "reliable": True,
                    },
                    "diagnostics": {
                        "publish_period_ms": 1000,
                        "qos": {
                            "depth": 10,
                            "reliable": True,
                        },
                    },
                }
            }
        },
    )

    env = os.environ.copy()
    processes = [
        subprocess.Popen(
            [
                ros2,
                "launch",
                "fa_resample",
                "fa_resample.launch.py",
                "node_name:=fa_resample_frame_buffer_integration",
                f"config_file:={resample_config}",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ),
        subprocess.Popen(
            [
                ros2,
                "launch",
                "fa_frame_buffer",
                "fa_frame_buffer.launch.py",
                "node_name:=fa_frame_buffer_integration",
                f"config_file:={frame_buffer_config}",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ),
    ]

    rclpy.init()
    node = rclpy.create_node("resample_frame_buffer_graph_test")
    publisher = node.create_publisher(AudioFrame, resample_input_topic, 10)
    received: list[AudioFrame] = []
    subscription = node.create_subscription(
        AudioFrame,
        chunk_topic,
        lambda msg: received.append(msg),
        10,
    )

    try:
        assert _wait_for_subscription(publisher, 8.0), _stop_processes(processes)
        assert _wait_for_publisher(node, chunk_topic, 8.0), _stop_processes(processes)

        frame = _audio_frame(_float32_payload(480))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received:
            publisher.publish(frame)
            rclpy.spin_once(node, timeout_sec=0.05)

        assert received, _stop_processes(processes)
        output = received[-1]
        assert output.source_id == "resample_frame_buffer_source"
        assert output.stream_id == "stream/resample_frame_buffer/chunk_16k"
        assert output.encoding == "FLOAT32LE"
        assert output.sample_rate == 16000
        assert output.channels == 1
        assert output.bit_depth == 32
        assert output.layout == "interleaved"
        assert output.epoch == 9
        assert len(output.data) == 160 * 4
        assert subscription.topic_name == chunk_topic
    finally:
        node.destroy_node()
        rclpy.shutdown()
        _stop_processes(processes)
