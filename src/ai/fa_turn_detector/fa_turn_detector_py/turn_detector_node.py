#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AudioFrame, TurnContext, TurnEnd, TurnEndRequest
from fa_turn_detector_py.backends.base import TurnDetectorBackend
from fa_turn_detector_py.backends.factory import (
    TurnDetectorBackendSettings,
    build_turn_detector_backend,
)


class FaTurnDetectorNode(Node):
    """Turn detection node using a local Smart Turn v3 ONNX model."""

    def __init__(self, *, parameter_overrides: Iterable[Parameter] | None = None) -> None:
        if parameter_overrides is None:
            super().__init__("fa_turn_detector")
        else:
            super().__init__(
                "fa_turn_detector",
                parameter_overrides=list(parameter_overrides),
            )

        self._declare_required_parameters()

        self.audio_topic = self._required_string_parameter("audio_topic")
        self.expected_stream_id = self._required_string_parameter("expected_stream_id")
        self.turn_context_topic = self._required_string_parameter("turn_context_topic")
        self.turn_end_request_topic = self._required_string_parameter(
            "turn_end_request_topic"
        )
        self.output_topic = self._required_string_parameter("output_topic")
        self.expected_source_id = self._required_string_parameter("expected_source_id")
        self._validate_identity_contract()

        self.backend = self._load_backend()
        self.audio_buffer: deque[float] = deque(maxlen=self.backend.sample_rate * 10)
        self._active_session_id = ""
        self._active_user_turn_id = 0
        self._context_active = False

        qos_audio = self._qos_profile(
            depth_parameter="audio.qos.depth",
            reliable_parameter="audio.qos.reliable",
        )
        qos_turn_context = self._qos_profile(
            depth_parameter="turn_context.qos.depth",
            reliable_parameter="turn_context.qos.reliable",
        )
        qos_turn_end_request = self._qos_profile(
            depth_parameter="turn_end_request.qos.depth",
            reliable_parameter="turn_end_request.qos.reliable",
        )
        qos_output = self._qos_profile(
            depth_parameter="output.qos.depth",
            reliable_parameter="output.qos.reliable",
        )

        self.turn_end_pub = self.create_publisher(TurnEnd, self.output_topic, qos_output)
        self.audio_sub = self.create_subscription(
            AudioFrame,
            self.audio_topic,
            self.on_audio,
            qos_audio,
        )
        self.turn_context_sub = self.create_subscription(
            TurnContext,
            self.turn_context_topic,
            self.on_turn_context,
            qos_turn_context,
        )
        self.turn_end_request_sub = self.create_subscription(
            TurnEndRequest,
            self.turn_end_request_topic,
            self.on_turn_end_request,
            qos_turn_end_request,
        )

        self.get_logger().info(
            "fa_turn_detector started: "
            f"audio={self.audio_topic} "
            f"turn_context={self.turn_context_topic} "
            f"turn_end_request={self.turn_end_request_topic} "
            f"output={self.output_topic} "
            f"expected_source_id={self.expected_source_id} "
            f"expected_stream_id={self.expected_stream_id} "
            f"backend.name={self.backend.name} "
            f"model={self.backend.model_path}"
        )

    def _declare_required_parameters(self) -> None:
        self.declare_parameter("audio_topic", Parameter.Type.STRING)
        self.declare_parameter("expected_stream_id", Parameter.Type.STRING)
        self.declare_parameter("turn_context_topic", Parameter.Type.STRING)
        self.declare_parameter("turn_end_request_topic", Parameter.Type.STRING)
        self.declare_parameter("output_topic", Parameter.Type.STRING)
        self.declare_parameter("expected_source_id", Parameter.Type.STRING)
        self.declare_parameter("backend.name", Parameter.Type.STRING)
        self.declare_parameter("backend.model_path", Parameter.Type.STRING)
        self.declare_parameter("backend.threshold", Parameter.Type.DOUBLE)
        self.declare_parameter("backend.execution_provider", Parameter.Type.STRING)
        self.declare_parameter("backend.command", Parameter.Type.STRING)
        self.declare_parameter("backend.args", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("backend.health_args", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("backend.timeout_sec", Parameter.Type.DOUBLE)
        self.declare_parameter("backend.workspace_dir", Parameter.Type.STRING)
        self.declare_parameter("backend.cleanup_audio_files", Parameter.Type.BOOL)
        self.declare_parameter("audio.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("audio.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("turn_context.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("turn_context.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("turn_end_request.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("turn_end_request.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("output.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("output.qos.reliable", Parameter.Type.BOOL)

    def _validate_identity_contract(self) -> None:
        for topic_name, topic_value in (
            ("audio_topic", self.audio_topic),
            ("turn_context_topic", self.turn_context_topic),
            ("turn_end_request_topic", self.turn_end_request_topic),
            ("output_topic", self.output_topic),
        ):
            if self._same_identity_string(self.expected_stream_id, topic_value):
                raise RuntimeError(
                    f"expected_stream_id must be distinct from ROS {topic_name}"
                )
        if self._same_identity_string(self.audio_topic, self.output_topic):
            raise RuntimeError("audio_topic must be distinct from output_topic")

    @staticmethod
    def _remove_leading_slashes(value: str) -> str:
        return value.lstrip("/")

    @classmethod
    def _same_identity_string(cls, left: str, right: str) -> bool:
        return left == right or (
            cls._remove_leading_slashes(left) == cls._remove_leading_slashes(right)
        )

    def _load_backend(self) -> TurnDetectorBackend:
        return build_turn_detector_backend(
            TurnDetectorBackendSettings(
                name=self._string_parameter("backend.name"),
                model_path=self._string_parameter("backend.model_path"),
                threshold=self._double_parameter("backend.threshold"),
                execution_provider=self._string_parameter("backend.execution_provider"),
                command=self._string_parameter("backend.command"),
                args=self._string_tuple_parameter("backend.args"),
                health_args=self._string_tuple_parameter("backend.health_args"),
                timeout_sec=self._double_parameter("backend.timeout_sec"),
                workspace_dir=self._string_parameter("backend.workspace_dir"),
                cleanup_audio_files=self._bool_parameter("backend.cleanup_audio_files"),
            )
        )

    def _string_tuple_parameter(self, name: str) -> tuple[str, ...]:
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.STRING_ARRAY:
            raise RuntimeError(f"{name} must be a string array")
        return tuple(parameter.get_parameter_value().string_array_value)

    def _string_parameter(self, name: str) -> str:
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.STRING:
            raise RuntimeError(f"{name} must be a string")
        return parameter.value

    def _required_string_parameter(self, name: str) -> str:
        value = FaTurnDetectorNode._string_parameter(self, name).strip()
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

    def _positive_integer_parameter(self, name: str) -> int:
        value = FaTurnDetectorNode._integer_parameter(self, name)
        if value <= 0:
            raise RuntimeError(f"{name} must be greater than zero")
        return value

    def _double_parameter(self, name: str) -> float:
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.DOUBLE:
            raise RuntimeError(f"{name} must be a double")
        return parameter.value

    def _qos_profile(self, *, depth_parameter: str, reliable_parameter: str) -> QoSProfile:
        depth = FaTurnDetectorNode._positive_integer_parameter(self, depth_parameter)
        reliable = FaTurnDetectorNode._bool_parameter(self, reliable_parameter)
        return FaTurnDetectorNode._qos_from_values(depth=depth, reliable=reliable)

    @staticmethod
    def _qos_from_values(*, depth: int, reliable: bool) -> QoSProfile:
        if depth <= 0:
            raise RuntimeError("qos depth must be greater than zero")
        qos = QoSProfile(depth=depth)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (
            ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
        )
        return qos

    def on_turn_context(self, msg: TurnContext) -> None:
        context_active = bool(msg.active) and bool(msg.session_id)
        if not context_active:
            self._active_session_id = ""
            self._active_user_turn_id = 0
            self._context_active = False
            self.audio_buffer.clear()
            return

        session_id = str(msg.session_id)
        user_turn_id = int(msg.user_turn_id)
        turn_changed = (
            not self._context_active
            or self._active_session_id != session_id
            or self._active_user_turn_id != user_turn_id
        )
        if turn_changed:
            self.audio_buffer.clear()

        self._active_session_id = session_id
        self._active_user_turn_id = user_turn_id
        self._context_active = True

    def on_audio(self, msg: AudioFrame) -> None:
        if not self._context_active:
            return
        try:
            if int(msg.sample_rate) != self.backend.sample_rate:
                raise ValueError(
                    "AudioFrame sample_rate must match backend sample_rate "
                    f"{self.backend.sample_rate}, got {msg.sample_rate}"
                )
            audio_data = self._frame_to_float(
                msg,
                expected_source_id=self.expected_source_id,
                expected_stream_id=self.expected_stream_id,
            )
        except ValueError as exc:
            self.get_logger().error(f"Dropping invalid AudioFrame: {exc}")
            return
        self.audio_buffer.extend(audio_data.tolist())

    def on_turn_end_request(self, msg: TurnEndRequest) -> None:
        if not self._context_active:
            return
        if msg.session_id != self._active_session_id:
            self.get_logger().debug("Dropping TurnEndRequest with stale session")
            return
        if int(msg.user_turn_id) != self._active_user_turn_id:
            self.get_logger().debug("Dropping TurnEndRequest with stale turn")
            return
        request_id = int(msg.request_id)
        if request_id <= 0:
            self.get_logger().warning("Dropping TurnEndRequest with invalid request_id")
            return
        self._detect_turn_end(request_id=request_id)

    def _detect_turn_end(self, *, request_id: int) -> None:
        if len(self.audio_buffer) < self.backend.min_samples:
            self.get_logger().debug(
                f"Not enough audio data for turn detection: {len(self.audio_buffer)} samples"
            )
            return

        available_samples = min(len(self.audio_buffer), self.backend.max_samples)
        audio = np.asarray(list(self.audio_buffer)[-available_samples:], dtype=np.float32)
        try:
            result = self.backend.detect(audio)
        except Exception as exc:
            self.get_logger().fatal(f"Turn detection backend failed: {exc}")
            rclpy.shutdown()
            raise

        out = TurnEnd()
        out.timestamp = self.get_clock().now().to_msg()
        out.session_id = self._active_session_id
        out.user_turn_id = int(self._active_user_turn_id)
        out.request_id = int(request_id)
        out.probability = float(result.probability)
        out.is_end = bool(result.is_end)
        self.turn_end_pub.publish(out)

        self.get_logger().info(
            "Turn end probability: "
            f"request_id={request_id} "
            f"probability={result.probability:.3f} "
            f"is_end={str(out.is_end).lower()}"
        )

    @staticmethod
    def _frame_to_float(
        msg: AudioFrame,
        *,
        expected_source_id: str,
        expected_stream_id: str,
    ) -> np.ndarray:
        if not msg.data:
            raise ValueError("AudioFrame data is required")
        if not msg.source_id or not msg.stream_id:
            raise ValueError("AudioFrame source_id and stream_id are required")
        if not expected_source_id:
            raise ValueError("expected_source_id is required")
        if not expected_stream_id:
            raise ValueError("expected_stream_id is required")
        if msg.source_id != expected_source_id:
            raise ValueError("AudioFrame source_id must match expected_source_id")
        if msg.stream_id != expected_stream_id:
            raise ValueError("AudioFrame stream_id must match expected_stream_id")
        if msg.layout != "interleaved":
            raise ValueError(f"AudioFrame layout must be interleaved, got {msg.layout}")
        if int(msg.channels) != 1:
            raise ValueError(f"AudioFrame channels must be 1, got {msg.channels}")
        if msg.encoding != "FLOAT32LE":
            raise ValueError(f"AudioFrame encoding must be FLOAT32LE, got {msg.encoding}")
        if int(msg.bit_depth) != 32:
            raise ValueError(f"AudioFrame bit_depth must be 32, got {msg.bit_depth}")
        if len(msg.data) % np.dtype("<f4").itemsize != 0:
            raise ValueError("AudioFrame float32 data length is not byte-aligned")
        samples = np.frombuffer(bytes(msg.data), dtype="<f4")
        if not np.all(np.isfinite(samples)):
            raise ValueError("AudioFrame contains non-finite samples")
        if np.any(samples < -1.0) or np.any(samples > 1.0):
            raise ValueError("AudioFrame samples must be normalized to [-1.0, 1.0]")
        return samples


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FaTurnDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError as exc:
        message = str(exc)
        if "context is not valid" not in message and "rcl_shutdown" not in message:
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
