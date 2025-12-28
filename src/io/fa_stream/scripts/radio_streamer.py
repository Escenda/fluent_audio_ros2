#!/usr/bin/env python3
"""
fv_audio radio streamer example.

This node subscribes to fv_audio/msg/AudioFrame and pipes the raw PCM samples
to an ffmpeg process that publishes to an Icecast/Shoutcast style URL.
"""

import os
import signal
import subprocess
import sys
from typing import Optional

import rclpy
from rclpy.node import Node

from fv_audio.msg import AudioFrame


class RadioStreamer(Node):
    def __init__(self) -> None:
        super().__init__("fv_audio_radio_streamer")

        self.declare_parameter("input_topic", "audio/frame")
        self.declare_parameter("ffmpeg_path", "ffmpeg")
        self.declare_parameter("output_url", "http://source:hackme@localhost:8000/live")
        self.declare_parameter("audio_codec", "libmp3lame")
        self.declare_parameter("bitrate", "128k")
        self.declare_parameter("container_format", "mp3")
        self.declare_parameter("content_type", "audio/mpeg")
        self.declare_parameter("loglevel", "warning")

        self._subscription = self.create_subscription(
            AudioFrame,
            self.get_parameter("input_topic").get_parameter_value().string_value,
            self._audio_callback,
            10,
        )
        self._ffmpeg_proc: Optional[subprocess.Popen[bytes]] = None
        self._expected_rate: Optional[int] = None
        self._expected_channels: Optional[int] = None

        self.get_logger().info(
            "Initialized radio streamer. Waiting for audio frames on %s",
            self._subscription.topic_name,
        )

    def destroy_node(self) -> bool:
        self._stop_ffmpeg()
        return super().destroy_node()

    def _audio_callback(self, msg: AudioFrame) -> None:
        if msg.bit_depth != 16:
            self.get_logger().throttle_error(
                5000, "Only 16-bit PCM is supported. Received %d-bit frame", msg.bit_depth
            )
            return

        if not self._ffmpeg_proc:
            self._start_ffmpeg(msg)

        if not self._ffmpeg_proc or self._ffmpeg_proc.stdin is None:
            return

        if self._expected_rate and msg.sample_rate != self._expected_rate:
            self.get_logger().warn(
                "Sample rate changed from %d to %d. Restarting ffmpeg.",
                self._expected_rate,
                msg.sample_rate,
            )
            self._stop_ffmpeg()
            self._start_ffmpeg(msg)

        if self._expected_channels and msg.channels != self._expected_channels:
            self.get_logger().warn(
                "Channel count changed from %d to %d. Restarting ffmpeg.",
                self._expected_channels,
                msg.channels,
            )
            self._stop_ffmpeg()
            self._start_ffmpeg(msg)

        if not self._ffmpeg_proc or self._ffmpeg_proc.stdin is None:
            return

        try:
            self._ffmpeg_proc.stdin.write(bytes(msg.data))
        except BrokenPipeError:
            self.get_logger().error("ffmpeg pipe closed unexpectedly. Restarting.")
            self._stop_ffmpeg()

    def _start_ffmpeg(self, msg: AudioFrame) -> None:
        ffmpeg_path = self.get_parameter("ffmpeg_path").get_parameter_value().string_value
        audio_codec = self.get_parameter("audio_codec").get_parameter_value().string_value
        bitrate = self.get_parameter("bitrate").get_parameter_value().string_value
        container_format = (
            self.get_parameter("container_format").get_parameter_value().string_value
        )
        content_type = self.get_parameter("content_type").get_parameter_value().string_value
        output_url = self.get_parameter("output_url").get_parameter_value().string_value
        loglevel = self.get_parameter("loglevel").get_parameter_value().string_value

        cmd = [
            ffmpeg_path,
            "-loglevel",
            loglevel,
            "-f",
            "s16le",
            "-ar",
            str(msg.sample_rate),
            "-ac",
            str(msg.channels),
            "-i",
            "pipe:0",
            "-c:a",
            audio_codec,
            "-b:a",
            bitrate,
            "-content_type",
            content_type,
            "-f",
            container_format,
            output_url,
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
        except FileNotFoundError:
            self.get_logger().error("ffmpeg binary '%s' not found.", ffmpeg_path)
            self._ffmpeg_proc = None

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
        except Exception:
            proc.kill()


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = RadioStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
