#!/usr/bin/env python3
"""
fa_stream network streaming sink.

This node subscribes to fa_interfaces/msg/AudioFrame and pipes the raw PCM samples
to an ffmpeg process that publishes to an Icecast/Shoutcast style URL.
"""

import sys

import rclpy
from rclpy.node import Node

from fa_interfaces.msg import AudioFrame
from fa_stream_py.backends.network_streamer import (
    AudioStreamFormat,
    NetworkStreamerBackend,
    NetworkStreamerConfig,
)


class FaStreamNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_stream")

        self.declare_parameter("input_topic")
        self.declare_parameter("ffmpeg_path")
        self.declare_parameter("output_url")
        self.declare_parameter("audio_codec")
        self.declare_parameter("bitrate")
        self.declare_parameter("container_format")
        self.declare_parameter("content_type")
        self.declare_parameter("loglevel")

        self._input_topic = self._required_string_parameter("input_topic")
        self._ffmpeg_path = self._required_string_parameter("ffmpeg_path")
        self._output_url = self._required_string_parameter("output_url")
        self._audio_codec = self._required_string_parameter("audio_codec")
        self._bitrate = self._required_string_parameter("bitrate")
        self._container_format = self._required_string_parameter("container_format")
        self._content_type = self._required_string_parameter("content_type")
        self._loglevel = self._required_string_parameter("loglevel")

        self._subscription = self.create_subscription(
            AudioFrame,
            self._input_topic,
            self._audio_callback,
            10,
        )
        self._streamer = NetworkStreamerBackend(
            NetworkStreamerConfig(
                ffmpeg_path=self._ffmpeg_path,
                output_url=self._output_url,
                audio_codec=self._audio_codec,
                bitrate=self._bitrate,
                container_format=self._container_format,
                content_type=self._content_type,
                loglevel=self._loglevel,
            )
        )

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
        self._streamer.close()
        return super().destroy_node()

    def _audio_callback(self, msg: AudioFrame) -> None:
        if not msg.source_id or not msg.stream_id:
            self.get_logger().error("AudioFrame source_id and stream_id are required")
            return
        if msg.layout != "interleaved":
            self.get_logger().error(
                "Only interleaved AudioFrame layout is supported. Received %s",
                msg.layout,
            )
            return
        if msg.encoding != "PCM16LE":
            self.get_logger().error(
                "Only PCM16LE AudioFrame encoding is supported. Received %s",
                msg.encoding,
            )
            return
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

        audio_format = AudioStreamFormat(
            sample_rate=msg.sample_rate,
            channels=msg.channels,
            encoding=msg.encoding,
            bit_depth=msg.bit_depth,
            layout=msg.layout,
        )
        self._streamer.ensure_started(audio_format)
        self._streamer.write(bytes(msg.data))


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
