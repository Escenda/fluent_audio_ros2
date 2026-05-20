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


def _pcm16_payload(sample_count: int) -> bytes:
    payload = bytearray()
    for index in range(sample_count):
        sample = (index % 128) - 64
        payload.extend(sample.to_bytes(2, byteorder="little", signed=True))
    return bytes(payload)


def _audio_frame(payload: bytes):
    from fa_interfaces.msg import AudioFrame

    msg = AudioFrame()
    msg.source_id = "sample_format_resample_source"
    msg.stream_id = "stream/sample_format_resample/raw_pcm16"
    msg.encoding = "PCM16LE"
    msg.sample_rate = 48000
    msg.channels = 1
    msg.bit_depth = 16
    msg.layout = "interleaved"
    msg.data = list(payload)
    msg.epoch = 7
    return msg


def test_sample_format_to_resample_launch_graph_publishes_16khz_float32(
    tmp_path: Path,
) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch graph integration")

    import rclpy
    from fa_interfaces.msg import AudioFrame

    raw_topic = "/audio/integration/sample_format_resample/raw_pcm16"
    float_topic = "/audio/integration/sample_format_resample/float32_48k"
    resampled_topic = "/audio/integration/sample_format_resample/float32_16k"

    sample_format_config = _write_yaml(
        tmp_path / "fa_sample_format.params.yaml",
        {
            "fa_sample_format_integration": {
                "ros__parameters": {
                    "input_topic": raw_topic,
                    "output_topic": float_topic,
                    "input_stream_id": "stream/sample_format_resample/raw_pcm16",
                    "input": {
                        "encoding": "PCM16LE",
                        "bit_depth": 16,
                    },
                    "output": {
                        "stream_id": "stream/sample_format_resample/float32_48k",
                        "encoding": "FLOAT32LE",
                        "bit_depth": 32,
                    },
                    "expected": {
                        "sample_rate": 48000,
                        "channels": 1,
                        "layout": "interleaved",
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
    resample_config = _write_yaml(
        tmp_path / "fa_resample.params.yaml",
        {
            "fa_resample_integration": {
                "ros__parameters": {
                    "target_sample_rate": 16000,
                    "backend": {
                        "name": "internal_linear_resampler",
                    },
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
                        "input_topic": float_topic,
                        "output_topic": resampled_topic,
                        "input_stream_id": "stream/sample_format_resample/float32_48k",
                        "output": {
                            "stream_id": "stream/sample_format_resample/float32_16k",
                        },
                    },
                    "ref": {
                        "enabled": False,
                        "input_topic": "/audio/integration/sample_format_resample/ref_in",
                        "output_topic": "/audio/integration/sample_format_resample/ref_out",
                        "input_stream_id": "stream/sample_format_resample/ref_float32_48k",
                        "output": {
                            "stream_id": "stream/sample_format_resample/ref_float32_16k",
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
                "fa_sample_format",
                "fa_sample_format.launch.py",
                "node_name:=fa_sample_format_integration",
                f"config_file:={sample_format_config}",
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
                "fa_resample",
                "fa_resample.launch.py",
                "node_name:=fa_resample_integration",
                f"config_file:={resample_config}",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ),
    ]

    rclpy.init()
    node = rclpy.create_node("sample_format_resample_graph_test")
    publisher = node.create_publisher(AudioFrame, raw_topic, 10)
    received: list[AudioFrame] = []
    subscription = node.create_subscription(
        AudioFrame,
        resampled_topic,
        lambda msg: received.append(msg),
        10,
    )

    try:
        assert _wait_for_subscription(publisher, 8.0), _stop_processes(processes)
        assert _wait_for_publisher(node, resampled_topic, 8.0), _stop_processes(processes)

        frame = _audio_frame(_pcm16_payload(480))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received:
            publisher.publish(frame)
            rclpy.spin_once(node, timeout_sec=0.05)

        assert received, _stop_processes(processes)
        output = received[-1]
        assert output.source_id == "sample_format_resample_source"
        assert output.stream_id == "stream/sample_format_resample/float32_16k"
        assert output.encoding == "FLOAT32LE"
        assert output.sample_rate == 16000
        assert output.channels == 1
        assert output.bit_depth == 32
        assert output.layout == "interleaved"
        assert output.epoch == 7
        assert len(output.data) == 160 * 4
    finally:
        node.destroy_node()
        rclpy.shutdown()
        _stop_processes(processes)
