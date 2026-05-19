#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Float32

from fa_interfaces.msg import AudioFrame, VadState
from fa_vad_py.backends.base import Float32MonoWindow
from fa_vad_py.backends.factory import VadBackendSettings, build_vad_backend
from fa_vad_py.contracts import (
    audio_frame_to_float_samples,
    validate_node_config,
    validate_qos_depth,
)


class FaVadNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_vad")
        self._declare_required_parameters()

        self._input_topic = self._required_string_parameter("input_topic")
        self._input_stream_id = self._required_string_parameter("input_stream_id")
        self._output_topic = self._required_string_parameter("output_topic")
        self._vad_state_topic = self._required_string_parameter("vad_state_topic")
        self._probability_topic = self._required_string_parameter("probability_topic")
        self._expected_source_id = self._required_string_parameter("expected_source_id")
        self._validate_identity_contract()

        self._publish_vad_state = self._bool_parameter("publish_vad_state")
        self._publish_probability = self._bool_parameter("publish_probability")
        self._log_events = self._bool_parameter("log_speech_events")

        self._target_sample_rate = self._integer_parameter("target_sample_rate")
        threshold_start = self._double_parameter("threshold_start")
        threshold_end = self._double_parameter("threshold_end")
        hangover_ms = self._integer_parameter("hangover_ms")
        backend_frame_ms = self._integer_parameter("backend.frame_ms")
        backend_window_samples = self._integer_parameter("backend.window_samples")
        backend_history_buffer_ms = self._integer_parameter("backend.history_buffer_ms")
        validate_node_config(
            target_sample_rate=self._target_sample_rate,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            hangover_ms=hangover_ms,
            backend_frame_ms=backend_frame_ms,
            backend_window_samples=backend_window_samples,
            backend_history_buffer_ms=backend_history_buffer_ms,
        )

        backend_name = self._string_parameter("backend.name").strip()
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
        self._vad_state_pub: rclpy.publisher.Publisher | None = None
        if self._publish_vad_state:
            self._vad_state_pub = self.create_publisher(VadState, self._vad_state_topic, qos)

        self._prob_pub: rclpy.publisher.Publisher | None = None
        if self._publish_probability:
            self._prob_pub = self.create_publisher(Float32, self._probability_topic, qos)

        self._audio_sub = self.create_subscription(
            AudioFrame, self._input_topic, self._on_audio_frame, qos
        )

        self._vad = build_vad_backend(
            VadBackendSettings(
                name=backend_name,
                sample_rate=self._target_sample_rate,
                frame_ms=backend_frame_ms,
                window_samples=backend_window_samples,
                history_buffer_ms=backend_history_buffer_ms,
                hangover_ms=hangover_ms,
                threshold_start=threshold_start,
                threshold_end=threshold_end,
                model_path=model_path,
                execution_provider=execution_provider,
                command=command,
                args=backend_args,
                timeout_sec=timeout_sec,
                workspace_dir=workspace_dir,
                cleanup_audio_files=cleanup_audio_files,
            )
        )

        self._last_is_speech: bool | None = None

        self.get_logger().info(
            "FA VAD: "
            f"input={self._input_topic} input_stream_id={self._input_stream_id} "
            f"expected_source_id={self._expected_source_id} "
            f"output={self._output_topic} "
            f"vad_state={self._vad_state_topic if self._publish_vad_state else '(disabled)'} "
            f"target_sr={self._target_sample_rate} start={threshold_start:.2f} "
            f"end={threshold_end:.2f} hangover={hangover_ms}ms "
            f"window_samples={backend_window_samples} "
            f"history_buffer={backend_history_buffer_ms}ms "
            f"backend.name={backend_name} provider={execution_provider} "
            f"model_path={model_path} command={command}"
        )

    def _declare_required_parameters(self) -> None:
        self.declare_parameter("input_topic", Parameter.Type.STRING)
        self.declare_parameter("input_stream_id", Parameter.Type.STRING)
        self.declare_parameter("output_topic", Parameter.Type.STRING)
        self.declare_parameter("vad_state_topic", Parameter.Type.STRING)
        self.declare_parameter("probability_topic", Parameter.Type.STRING)
        self.declare_parameter("expected_source_id", Parameter.Type.STRING)
        self.declare_parameter("publish_vad_state", Parameter.Type.BOOL)
        self.declare_parameter("publish_probability", Parameter.Type.BOOL)

        self.declare_parameter("target_sample_rate", Parameter.Type.INTEGER)
        self.declare_parameter("threshold_start", Parameter.Type.DOUBLE)
        self.declare_parameter("threshold_end", Parameter.Type.DOUBLE)
        self.declare_parameter("hangover_ms", Parameter.Type.INTEGER)

        self.declare_parameter("backend.name", Parameter.Type.STRING)
        self.declare_parameter("backend.frame_ms", Parameter.Type.INTEGER)
        self.declare_parameter("backend.window_samples", Parameter.Type.INTEGER)
        self.declare_parameter("backend.history_buffer_ms", Parameter.Type.INTEGER)
        self.declare_parameter("backend.model_path", Parameter.Type.STRING)
        self.declare_parameter("backend.execution_provider", Parameter.Type.STRING)
        self.declare_parameter("backend.command", Parameter.Type.STRING)
        self.declare_parameter("backend.args", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("backend.timeout_sec", Parameter.Type.DOUBLE)
        self.declare_parameter("backend.workspace_dir", Parameter.Type.STRING)
        self.declare_parameter("backend.cleanup_audio_files", Parameter.Type.BOOL)

        self.declare_parameter("log_speech_events", Parameter.Type.BOOL)

        self.declare_parameter("qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("qos.reliable", Parameter.Type.BOOL)

    def _validate_identity_contract(self) -> None:
        if self._same_identity_string(self._input_stream_id, self._input_topic):
            raise RuntimeError("input_stream_id must be distinct from ROS input_topic")
        if self._same_identity_string(self._input_stream_id, self._output_topic):
            raise RuntimeError("input_stream_id must be distinct from ROS output_topic")
        if self._same_identity_string(self._input_stream_id, self._vad_state_topic):
            raise RuntimeError("input_stream_id must be distinct from ROS vad_state_topic")
        if self._same_identity_string(self._input_stream_id, self._probability_topic):
            raise RuntimeError("input_stream_id must be distinct from ROS probability_topic")
        if self._same_identity_string(self._input_topic, self._output_topic):
            raise RuntimeError("input_topic must be distinct from output_topic")

    @staticmethod
    def _remove_leading_slashes(value: str) -> str:
        return value.lstrip("/")

    @classmethod
    def _same_identity_string(cls, left: str, right: str) -> bool:
        return left == right or (
            cls._remove_leading_slashes(left) == cls._remove_leading_slashes(right)
        )

    def _string_parameter(self, name: str) -> str:
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.STRING:
            raise RuntimeError(f"{name} must be a string")
        return parameter.value

    def _required_string_parameter(self, name: str) -> str:
        value = self._string_parameter(name).strip()
        if not value:
            raise RuntimeError(f"{name} is required")
        return value

    def _bool_parameter(self, name: str) -> bool:
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.BOOL:
            raise RuntimeError(f"{name} must be a bool")
        return parameter.value

    def _integer_parameter(self, name: str) -> int:
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.INTEGER:
            raise RuntimeError(f"{name} must be an integer")
        return parameter.value

    def _double_parameter(self, name: str) -> float:
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.DOUBLE:
            raise RuntimeError(f"{name} must be a double")
        return parameter.value

    def _string_array_parameter(self, name: str) -> tuple[str, ...]:
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
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
            out.source_id = msg.source_id
            out.stream_id = msg.stream_id
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
            expected_source_id=self._expected_source_id,
            expected_stream_id=self._input_stream_id,
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
