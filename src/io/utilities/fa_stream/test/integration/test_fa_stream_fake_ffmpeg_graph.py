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


def _write_fake_ffmpeg(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import os",
                "import sys",
                "",
                "argv_path = Path(os.environ['FA_STREAM_FAKE_FFMPEG_ARGV'])",
                "stdin_path = Path(os.environ['FA_STREAM_FAKE_FFMPEG_STDIN'])",
                "argv_path.write_text('\\n'.join(sys.argv[1:]), encoding='utf-8')",
                "with stdin_path.open('ab') as stream:",
                "    while True:",
                "        chunk = os.read(sys.stdin.fileno(), 4096)",
                "        if not chunk:",
                "            break",
                "        stream.write(chunk)",
                "        stream.flush()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
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


def _file_contains_payload(path: Path, payload: bytes) -> bool:
    if not path.exists():
        return False
    return payload in path.read_bytes()


def _assert_payload_chunks(streamed: bytes, payload: bytes) -> None:
    assert streamed
    assert len(streamed) % len(payload) == 0
    for offset in range(0, len(streamed), len(payload)):
        assert streamed[offset : offset + len(payload)] == payload


def _audio_frame(payload: bytes):
    from fa_interfaces.msg import AudioFrame

    msg = AudioFrame()
    msg.source_id = "fa_stream_integration_source"
    msg.stream_id = "fa_stream_integration_stream"
    msg.encoding = "PCM16LE"
    msg.sample_rate = 16000
    msg.channels = 1
    msg.bit_depth = 16
    msg.layout = "interleaved"
    msg.data = list(payload)
    msg.epoch = 1
    return msg


def test_fa_stream_pipes_pcm16le_frames_to_configured_ffmpeg(tmp_path: Path) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for fa_stream graph integration")

    import rclpy
    from fa_interfaces.msg import AudioFrame

    input_topic = "audio/integration/fa_stream_input"
    fake_ffmpeg = _write_fake_ffmpeg(tmp_path / "fake_ffmpeg")
    fake_argv = tmp_path / "ffmpeg_argv.txt"
    fake_stdin = tmp_path / "ffmpeg_stdin.pcm"
    config_path = _write_yaml(
        tmp_path / "fa_stream.params.yaml",
        {
            "fa_stream_integration": {
                "ros__parameters": {
                    "input_topic": input_topic,
                    "ffmpeg_path": str(fake_ffmpeg),
                    "output_url": "test://sink",
                    "audio_codec": "pcm_s16le",
                    "bitrate": "128k",
                    "container_format": "s16le",
                    "content_type": "audio/x-test",
                    "loglevel": "error",
                }
            }
        },
    )

    env = os.environ.copy()
    env["FA_STREAM_FAKE_FFMPEG_ARGV"] = str(fake_argv)
    env["FA_STREAM_FAKE_FFMPEG_STDIN"] = str(fake_stdin)

    process = subprocess.Popen(
        [
            ros2,
            "launch",
            "fa_stream",
            "fa_stream.launch.py",
            "node_name:=fa_stream_integration",
            f"config_file:={config_path}",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    rclpy.init()
    node = rclpy.create_node("fa_stream_graph_test")
    publisher = node.create_publisher(AudioFrame, input_topic, 10)
    payload = b"\x01\x00\x02\x00\x03\x00\x04\x00"

    try:
        assert _wait_for_subscription(publisher, 8.0), _stop_process(process)

        deadline = time.monotonic() + 5.0
        while (
            time.monotonic() < deadline
            and not _file_contains_payload(fake_stdin, payload)
        ):
            publisher.publish(_audio_frame(payload))
            rclpy.spin_once(node, timeout_sec=0.05)

        _assert_payload_chunks(fake_stdin.read_bytes(), payload)
        argv = fake_argv.read_text(encoding="utf-8").splitlines()
        assert argv == [
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_s16le",
            "-b:a",
            "128k",
            "-content_type",
            "audio/x-test",
            "-f",
            "s16le",
            "test://sink",
        ]
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if process.poll() is None:
            _stop_process(process)
