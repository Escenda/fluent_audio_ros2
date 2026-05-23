#!/usr/bin/env python3
from __future__ import annotations

from array import array

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AsrControl, AudioFrame
from fa_dialogue_py.wake_ack_tone import WakeAckToneConfig, synthesize_wake_ack_pcm16


class FaWakeAckNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_wake_ack")
        self._declare_parameters()
        self.enabled = self._bool_parameter("enabled")
        self.control_topic = self._required_string_parameter("control_topic")
        self.output_topic = self._required_string_parameter("output_topic")
        self.output_source_id = self._required_string_parameter("output.source_id")
        self.output_stream_id = self._required_string_parameter("output.stream_id")
        self.cooldown_ms = self._non_negative_integer_parameter("cooldown_ms")
        self._last_publish_ms: int | None = None

        self.tone_config = WakeAckToneConfig(
            sample_rate=self._positive_integer_parameter("audio.sample_rate"),
            channels=self._positive_integer_parameter("audio.channels"),
            duration_ms=self._positive_integer_parameter("tone.duration_ms"),
            fade_ms=self._non_negative_integer_parameter("tone.fade_ms"),
            gain=self._float_parameter("tone.gain"),
            base_hz=self._float_parameter("tone.base_hz"),
            lift_hz=self._float_parameter("tone.lift_hz"),
            shimmer_hz=self._float_parameter("tone.shimmer_hz"),
        )
        self._audio_bytes = synthesize_wake_ack_pcm16(self.tone_config)

        control_qos = self._qos_profile("control.qos.depth", "control.qos.reliable")
        audio_qos = self._qos_profile("output.qos.depth", "output.qos.reliable")
        self.audio_pub = self.create_publisher(AudioFrame, self.output_topic, audio_qos)
        self.control_sub = self.create_subscription(
            AsrControl,
            self.control_topic,
            self.on_control,
            control_qos,
        )
        self.get_logger().info(
            "fa_wake_ack started: "
            f"control={self.control_topic} output={self.output_topic} "
            f"stream={self.output_stream_id} enabled={self.enabled}"
        )

    def on_control(self, msg: AsrControl) -> None:
        if not self.enabled:
            return
        if msg.action != AsrControl.ACTION_START or msg.reason != "wake":
            return
        now_ms = self._now_ms()
        if self._last_publish_ms is not None and now_ms - self._last_publish_ms < self.cooldown_ms:
            return
        self._last_publish_ms = now_ms
        self.audio_pub.publish(self._build_frame())
        self.get_logger().info(
            f"Wake ack tone published: session={msg.session_id} turn={msg.user_turn_id}"
        )

    def _build_frame(self) -> AudioFrame:
        frame = AudioFrame()
        frame.header.stamp = self.get_clock().now().to_msg()
        frame.source_id = self.output_source_id
        frame.stream_id = self.output_stream_id
        frame.encoding = "PCM16LE"
        frame.sample_rate = self.tone_config.sample_rate
        frame.channels = self.tone_config.channels
        frame.bit_depth = 16
        frame.layout = "interleaved"
        frame.data = array("B", self._audio_bytes)
        frame.epoch = 0
        return frame

    def _declare_parameters(self) -> None:
        self.declare_parameter("enabled", Parameter.Type.BOOL)
        self.declare_parameter("control_topic", Parameter.Type.STRING)
        self.declare_parameter("output_topic", Parameter.Type.STRING)
        self.declare_parameter("output.source_id", Parameter.Type.STRING)
        self.declare_parameter("output.stream_id", Parameter.Type.STRING)
        self.declare_parameter("cooldown_ms", Parameter.Type.INTEGER)
        self.declare_parameter("audio.sample_rate", Parameter.Type.INTEGER)
        self.declare_parameter("audio.channels", Parameter.Type.INTEGER)
        self.declare_parameter("tone.duration_ms", Parameter.Type.INTEGER)
        self.declare_parameter("tone.fade_ms", Parameter.Type.INTEGER)
        self.declare_parameter("tone.gain", Parameter.Type.DOUBLE)
        self.declare_parameter("tone.base_hz", Parameter.Type.DOUBLE)
        self.declare_parameter("tone.lift_hz", Parameter.Type.DOUBLE)
        self.declare_parameter("tone.shimmer_hz", Parameter.Type.DOUBLE)
        self.declare_parameter("control.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("control.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("output.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("output.qos.reliable", Parameter.Type.BOOL)

    def _required_string_parameter(self, name: str) -> str:
        value = self.get_parameter(name).get_parameter_value().string_value.strip()
        if not value:
            raise RuntimeError(f"{name} is required")
        return value

    def _bool_parameter(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def _positive_integer_parameter(self, name: str) -> int:
        value = self.get_parameter(name).get_parameter_value().integer_value
        if value <= 0:
            raise RuntimeError(f"{name} must be positive")
        return value

    def _non_negative_integer_parameter(self, name: str) -> int:
        value = self.get_parameter(name).get_parameter_value().integer_value
        if value < 0:
            raise RuntimeError(f"{name} must be >= 0")
        return value

    def _float_parameter(self, name: str) -> float:
        value = self.get_parameter(name).get_parameter_value().double_value
        if value <= 0.0:
            raise RuntimeError(f"{name} must be positive")
        return float(value)

    def _qos_profile(self, depth_parameter: str, reliable_parameter: str) -> QoSProfile:
        depth = self._positive_integer_parameter(depth_parameter)
        reliable = self._bool_parameter(reliable_parameter)
        profile = QoSProfile(depth=depth)
        profile.history = HistoryPolicy.KEEP_LAST
        profile.reliability = ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
        return profile

    def _now_ms(self) -> int:
        now = self.get_clock().now().to_msg()
        return (now.sec * 1000) + (now.nanosec // 1_000_000)


def main() -> None:
    rclpy.init()
    node: FaWakeAckNode | None = None
    executor: SingleThreadedExecutor | None = None
    try:
        node = FaWakeAckNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
