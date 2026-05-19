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


def _pcm16_payload(sample: int, sample_count: int) -> bytes:
    payload = bytearray()
    for _index in range(sample_count):
        payload.extend(sample.to_bytes(2, byteorder="little", signed=True))
    return bytes(payload)


def _audio_frame(
    *,
    source_id: str,
    stream_id: str,
    payload: bytes,
    stamp,
    epoch: int,
):
    from fa_interfaces.msg import AudioFrame

    msg = AudioFrame()
    msg.header.stamp = stamp
    msg.source_id = source_id
    msg.stream_id = stream_id
    msg.encoding = "PCM16LE"
    msg.sample_rate = 16000
    msg.channels = 1
    msg.bit_depth = 16
    msg.layout = "interleaved"
    msg.data = list(payload)
    msg.epoch = epoch
    return msg


def test_fa_mix_two_input_launch_graph_mixes_only_when_inputs_are_ready(
    tmp_path: Path,
) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for fa_mix graph integration")

    import rclpy
    from fa_interfaces.msg import AudioFrame

    master_topic = "/audio/integration/fa_mix/master"
    aux_topic = "/audio/integration/fa_mix/aux"
    output_topic = "/audio/integration/fa_mix/output"
    config_path = _write_yaml(
        tmp_path / "fa_mix.params.yaml",
        {
            "fa_mix_integration": {
                "ros__parameters": {
                    "input_topics": [master_topic, aux_topic],
                    "input_stream_ids": [
                        "stream/fa_mix/master",
                        "stream/fa_mix/aux",
                    ],
                    "input_gains_db": [0.0, 0.0],
                    "master_index": 0,
                    "output_topic": output_topic,
                    "output": {
                        "stream_id": "stream/fa_mix/output",
                    },
                    "expected": {
                        "sample_rate": 16000,
                        "channels": 1,
                        "bit_depth": 16,
                        "encoding": "PCM16LE",
                        "layout": "interleaved",
                    },
                    "max_frame_age_ms": 1000,
                    "qos": {
                        "depth": 10,
                        "reliable": True,
                    },
                    "diagnostics": {
                        "qos": {
                            "depth": 10,
                            "reliable": True,
                        },
                        "publish_period_ms": 1000,
                    },
                }
            }
        },
    )

    process = subprocess.Popen(
        [
            ros2,
            "launch",
            "fa_mix",
            "fa_mix.launch.py",
            "node_name:=fa_mix_integration",
            f"config_file:={config_path}",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    rclpy.init()
    node = rclpy.create_node("fa_mix_two_input_graph_test")
    master_pub = node.create_publisher(AudioFrame, master_topic, 10)
    aux_pub = node.create_publisher(AudioFrame, aux_topic, 10)
    received: list[AudioFrame] = []
    subscription = node.create_subscription(
        AudioFrame,
        output_topic,
        lambda msg: received.append(msg),
        10,
    )

    try:
        assert _wait_for_subscription(master_pub, 8.0), _stop_process(process)
        assert _wait_for_subscription(aux_pub, 8.0), _stop_process(process)
        assert _wait_for_publisher(node, output_topic, 8.0), _stop_process(process)

        stamp = node.get_clock().now().to_msg()
        master_frame = _audio_frame(
            source_id="fa_mix_master_source",
            stream_id="stream/fa_mix/master",
            payload=_pcm16_payload(1000, 4),
            stamp=stamp,
            epoch=2,
        )
        aux_frame = _audio_frame(
            source_id="fa_mix_aux_source",
            stream_id="stream/fa_mix/aux",
            payload=_pcm16_payload(2000, 4),
            stamp=stamp,
            epoch=3,
        )

        master_pub.publish(master_frame)
        no_output_deadline = time.monotonic() + 0.25
        while time.monotonic() < no_output_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        assert not received

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received:
            aux_pub.publish(aux_frame)
            master_pub.publish(master_frame)
            rclpy.spin_once(node, timeout_sec=0.05)

        assert received, _stop_process(process)
        output = received[-1]
        assert output.source_id == "fa_mix_master_source"
        assert output.stream_id == "stream/fa_mix/output"
        assert output.encoding == "PCM16LE"
        assert output.sample_rate == 16000
        assert output.channels == 1
        assert output.bit_depth == 16
        assert output.layout == "interleaved"
        assert output.epoch == 3
        assert bytes(output.data) == _pcm16_payload(3000, 4)
        assert subscription.topic_name == output_topic
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if process.poll() is None:
            _stop_process(process)
