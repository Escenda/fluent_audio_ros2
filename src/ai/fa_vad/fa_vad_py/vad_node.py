#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable, Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Float32

from fa_interfaces.msg import AudioFrame, VadState
from fa_vad_py.backends.base import Float32MonoWindow, VADBackend
from fa_vad_py.backends.silero import SileroVAD
from fa_vad_py.contracts import (
    audio_frame_to_float_samples,
    validate_node_config,
    validate_qos_depth,
)


class FaVadNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_vad_node")

        self.declare_parameter("input_topic", "audio/frame")
        self.declare_parameter("output_topic", "audio/vad")
        self.declare_parameter("vad_state_topic", "voice/vad_state")
        self.declare_parameter("probability_topic", "audio/vad/probability")
        self.declare_parameter("publish_vad_state", True)
        self.declare_parameter("publish_probability", False)

        self.declare_parameter("target_sample_rate", 16000)
        self.declare_parameter("threshold_start", 0.5)
        self.declare_parameter("threshold_end", 0.1)
        self.declare_parameter("hangover_ms", 300)

        self.declare_parameter("backend.name", "")
        self.declare_parameter("backend.model_path", "")
        self.declare_parameter("backend.execution_provider", "")
        self.declare_parameter("backend.command", "")
        self.declare_parameter(
            "backend.args",
            Parameter.Type.STRING_ARRAY,
        )
        self.declare_parameter("backend.timeout_sec", 1.0)
        self.declare_parameter("backend.workspace_dir", "/tmp/fluent_audio/fa_vad")
        self.declare_parameter("backend.cleanup_audio_files", True)

        self.declare_parameter("log_speech_events", True)

        self.declare_parameter("qos.depth", 10)
        self.declare_parameter("qos.reliable", False)

        self._input_topic = self._string_parameter("input_topic")
        self._output_topic = self._string_parameter("output_topic")
        self._vad_state_topic = self._string_parameter("vad_state_topic")
        self._probability_topic = self._string_parameter("probability_topic")

        self._publish_vad_state = self._bool_parameter("publish_vad_state")
        self._publish_probability = self._bool_parameter("publish_probability")
        self._log_events = self._bool_parameter("log_speech_events")

        self._target_sample_rate = self._integer_parameter("target_sample_rate")
        threshold_start = self._double_parameter("threshold_start")
        threshold_end = self._double_parameter("threshold_end")
        hangover_ms = self._integer_parameter("hangover_ms")
        validate_node_config(
            target_sample_rate=self._target_sample_rate,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            hangover_ms=hangover_ms,
        )

        model_path = self._string_parameter("backend.model_path").strip()
        execution_provider = self._string_parameter("backend.execution_provider").strip()
        command = self._string_parameter("backend.command").strip()
        backend_args = self._string_array_parameter("backend.args")
        timeout_sec = self._double_parameter("backend.timeout_sec")
        workspace_dir = self._string_parameter("backend.workspace_dir").strip()
        cleanup_audio_files = self._bool_parameter("backend.cleanup_audio_files")

        depth = validate_qos_depth(self._integer_parameter("qos.depth"))
        reliable = self._bool_parameter("qos.reliable")

        qos = QoSProfile(depth=depth)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (
            ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
        )

        self._vad_pub = self.create_publisher(Bool, self._output_topic, qos)
        self._vad_state_pub: Optional[rclpy.publisher.Publisher] = None
        if self._publish_vad_state:
            self._vad_state_pub = self.create_publisher(VadState, self._vad_state_topic, qos)

        self._prob_pub: Optional[rclpy.publisher.Publisher] = None
        if self._publish_probability:
            self._prob_pub = self.create_publisher(Float32, self._probability_topic, qos)

        self._audio_sub = self.create_subscription(
            AudioFrame, self._input_topic, self._on_audio_frame, qos
        )

        self._vad = self._load_backend(
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            hangover_ms=hangover_ms,
            model_path=model_path,
            execution_provider=execution_provider,
            command=command,
            args=backend_args,
            timeout_sec=timeout_sec,
            workspace_dir=workspace_dir,
            cleanup_audio_files=cleanup_audio_files,
        )

        self._last_is_speech: Optional[bool] = None

        self.get_logger().info(
            "FA VAD (Silero): "
            f"input={self._input_topic} output={self._output_topic} "
            f"vad_state={self._vad_state_topic if self._publish_vad_state else '(disabled)'} "
            f"target_sr={self._target_sample_rate} start={threshold_start:.2f} "
            f"end={threshold_end:.2f} hangover={hangover_ms}ms "
            f"provider={execution_provider} model_path={model_path} command={command}"
        )

    def _load_backend(
        self,
        *,
        threshold_start: float,
        threshold_end: float,
        hangover_ms: int,
        model_path: str,
        execution_provider: str,
        command: str,
        args: tuple[str, ...],
        timeout_sec: float,
        workspace_dir: str,
        cleanup_audio_files: bool,
    ) -> VADBackend:
        backend_name = self._string_parameter("backend.name").strip()
        if not backend_name:
            raise RuntimeError("backend.name is required")
        if backend_name != SileroVAD.name:
            raise RuntimeError(f"unsupported VAD backend.name: {backend_name}")
        return SileroVAD(
            sample_rate=self._target_sample_rate,
            frame_ms=20,
            hangover_ms=hangover_ms,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            model_path=model_path,
            execution_provider=execution_provider,
            command=command,
            args=args,
            timeout_sec=timeout_sec,
            workspace_dir=workspace_dir,
            cleanup_audio_files=cleanup_audio_files,
        )

    def _string_parameter(self, name: str) -> str:
        parameter = self.get_parameter(name)
        if parameter.type_ != Parameter.Type.STRING:
            raise RuntimeError(f"{name} must be a string")
        return parameter.value

    def _bool_parameter(self, name: str) -> bool:
        parameter = self.get_parameter(name)
        if parameter.type_ != Parameter.Type.BOOL:
            raise RuntimeError(f"{name} must be a bool")
        return parameter.value

    def _integer_parameter(self, name: str) -> int:
        parameter = self.get_parameter(name)
        if parameter.type_ != Parameter.Type.INTEGER:
            raise RuntimeError(f"{name} must be an integer")
        return parameter.value

    def _double_parameter(self, name: str) -> float:
        parameter = self.get_parameter(name)
        if parameter.type_ != Parameter.Type.DOUBLE:
            raise RuntimeError(f"{name} must be a double")
        return parameter.value

    def _string_array_parameter(self, name: str) -> tuple[str, ...]:
        parameter = self.get_parameter(name)
        if parameter.type_ != Parameter.Type.STRING_ARRAY:
            raise RuntimeError(f"{name} must be a string array")
        return tuple(parameter.get_parameter_value().string_array_value)

    def _on_audio_frame(self, msg: AudioFrame) -> None:
        try:
            if int(msg.sample_rate) != self._target_sample_rate:
                raise ValueError(
                    "AudioFrame sample_rate must match target_sample_rate "
                    f"{self._target_sample_rate}, got {msg.sample_rate}"
                )
            window = self._audio_to_window(msg)
        except ValueError as exc:
            self.get_logger().error(f"Dropping invalid AudioFrame: {exc}")
            return
        try:
            decision = self._vad.update(window)
        except Exception as exc:
            self.get_logger().fatal(f"VAD backend failed: {exc}")
            rclpy.shutdown()
            raise
        if decision is None:
            return
        probability, is_speech, start, end = decision

        if self._log_events:
            if start:
                self.get_logger().info(f"Speech START (prob={probability:.2f})")
            if end:
                self.get_logger().info(f"Speech END (prob={probability:.2f})")

        if self._vad_state_pub is not None:
            out = VadState()
            out.header = msg.header
            out.probability = float(probability)
            out.is_speech = bool(is_speech)
            out.start = bool(start)
            out.end = bool(end)
            self._vad_state_pub.publish(out)

        if self._prob_pub is not None:
            prob_msg = Float32()
            prob_msg.data = float(probability)
            self._prob_pub.publish(prob_msg)

        if self._last_is_speech is None or self._last_is_speech != bool(is_speech):
            self._last_is_speech = bool(is_speech)
            vad_msg = Bool()
            vad_msg.data = self._last_is_speech
            self._vad_pub.publish(vad_msg)

    def _audio_to_window(self, msg: AudioFrame) -> Float32MonoWindow:
        data = bytes(msg.data)
        audio_frame_to_float_samples(
            data=data,
            source_id=msg.source_id,
            stream_id=msg.stream_id,
            expected_stream_id=self._input_topic,
            encoding=msg.encoding,
            layout=msg.layout,
            channels=int(msg.channels),
            bit_depth=int(msg.bit_depth),
        )
        return Float32MonoWindow(sample_rate=int(msg.sample_rate), data=data)


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FaVadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
