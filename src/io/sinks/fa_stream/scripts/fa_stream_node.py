#!/usr/bin/env python3
"""
fa_stream network streaming sink.

This node subscribes to fa_interfaces/msg/AudioFrame and pipes the raw PCM samples
to an ffmpeg process that publishes to an Icecast/Shoutcast style URL.
"""

import shutil
import signal
import subprocess
import sys

import rclpy
from rclpy.node import Node

from fa_interfaces.msg import AudioFrame


class FaStreamNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_stream")

        self.declare_parameter("input_topic", "audio/frame")
        self.declare_parameter("ffmpeg_path", "ffmpeg")
        self.declare_parameter("output_url", "")
        self.declare_parameter("audio_codec", "libmp3lame")
        self.declare_parameter("bitrate", "128k")
        self.declare_parameter("container_format", "mp3")
        self.declare_parameter("content_type", "audio/mpeg")
        self.declare_parameter("loglevel", "warning")

        self._input_topic = self._required_string_parameter("input_topic")
        self._ffmpeg_path = self._required_string_parameter("ffmpeg_path")
        self._output_url = self._required_string_parameter("output_url")
        self._audio_codec = self._required_string_parameter("audio_codec")
        self._bitrate = self._required_string_parameter("bitrate")
        self._container_format = self._required_string_parameter("container_format")
        self._content_type = self._required_string_parameter("content_type")
        self._loglevel = self._required_string_parameter("loglevel")

        if shutil.which(self._ffmpeg_path) is None:
            raise RuntimeError(f"ffmpeg_path is not executable: {self._ffmpeg_path}")

        self._subscription = self.create_subscription(
            AudioFrame,
            self._input_topic,
            self._audio_callback,
            10,
        )
        self._ffmpeg_proc: subprocess.Popen[bytes] | None = None
        self._expected_rate: int | None = None
        self._expected_channels: int | None = None

        self.get_logger().info(
            "Initialized FA Stream. Waiting for audio frames on %s",
            self._subscription.topic_name,
        )

    def _required_string_parameter(self, name: str) -> str:
        value = self.get_parameter(name).get_parameter_value().string_value.strip()
        if not value:
            raise RuntimeError(f"{name} is required")
        return value

    def destroy_node(self) -> bool:
        self._stop_ffmpeg()
        return super().destroy_node()

    def _audio_callback(self, msg: AudioFrame) -> None:
        if msg.bit_depth != 16:
            self.get_logger().error(
                "Only 16-bit PCM is supported. Received %d-bit frame", msg.bit_depth
            )
            return
        if msg.sample_rate == 0 or msg.channels == 0 or not msg.data:
            self.get_logger().error(
                "Invalid audio frame: sample_rate=%d channels=%d bytes=%d",
                msg.sample_rate,
                msg.channels,
                len(msg.data),
            )
            return

        if not self._ffmpeg_proc:
            self._start_ffmpeg(msg)

        if not self._ffmpeg_proc or self._ffmpeg_proc.stdin is None:
            raise RuntimeError("ffmpeg process is not available after start")

        if self._expected_rate and msg.sample_rate != self._expected_rate:
            raise RuntimeError(
                "Sample rate changed during stream: "
                f"{self._expected_rate} -> {msg.sample_rate}"
            )

        if self._expected_channels and msg.channels != self._expected_channels:
            raise RuntimeError(
                "Channel count changed during stream: "
                f"{self._expected_channels} -> {msg.channels}"
            )

        try:
            self._ffmpeg_proc.stdin.write(bytes(msg.data))
        except BrokenPipeError as exc:
            self._stop_ffmpeg()
            raise RuntimeError("ffmpeg pipe closed unexpectedly") from exc

    def _start_ffmpeg(self, msg: AudioFrame) -> None:
        cmd = [
            self._ffmpeg_path,
            "-loglevel",
            self._loglevel,
            "-f",
            "s16le",
            "-ar",
            str(msg.sample_rate),
            "-ac",
            str(msg.channels),
            "-i",
            "pipe:0",
            "-c:a",
            self._audio_codec,
            "-b:a",
            self._bitrate,
            "-content_type",
            self._content_type,
            "-f",
            self._container_format,
            self._output_url,
        ]

        self.get_logger().info("Starting ffmpeg: %s", " ".join(cmd))
        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._expected_rate = msg.sample_rate
            self._expected_channels = msg.channels
        except OSError as exc:
            self._ffmpeg_proc = None
            raise RuntimeError(f"failed to start ffmpeg: {exc}") from exc

    def _stop_ffmpeg(self) -> None:
        if not self._ffmpeg_proc:
            return

        proc = self._ffmpeg_proc
        self._ffmpeg_proc = None
        self._expected_rate = None
        self._expected_channels = None

        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass

        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = FaStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
