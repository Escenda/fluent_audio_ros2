#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AudioFrame, TurnContext, TurnEnd, VadState
from fa_turn_detector_py.backends.base import TurnDetectorBackend
from fa_turn_detector_py.backends.factory import (
    TurnDetectorBackendSettings,
    build_turn_detector_backend,
)


@dataclass(frozen=True)
class ControlInputConfig:
    control_id: str
    action: str
    topic: str
    msg_type: str
    source_id: str
    stream_id: str
    active_field: str
    start_field: str
    end_field: str
    close_on: str
    qos_depth: int
    qos_reliable: bool


@dataclass(frozen=True)
class ControlEvent:
    control_id: str
    source_id: str
    stream_id: str
    active: bool
    start: bool
    end: bool


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
        self.output_topic = self._required_string_parameter("output_topic")
        self.expected_source_id = self._required_string_parameter("expected_source_id")
        self.control_default_enabled = self._bool_parameter("control.default_enabled")
        self._validate_control_default_enabled()
        self.control_config = self._load_control_config()
        self._validate_identity_contract()
        self._validate_control_binding_contract()

        self.backend = self._load_backend()
        self.audio_buffer: deque[float] = deque(maxlen=self.backend.sample_rate * 10)
        self.is_speech = False
        self._active_session_id = ""
        self._active_user_turn_id = 0
        self._context_active = False

        qos_audio = self._qos_profile(
            depth_parameter="audio.qos.depth",
            reliable_parameter="audio.qos.reliable",
        )
        qos_control = self._qos_from_values(
            depth=self.control_config.qos_depth,
            reliable=self.control_config.qos_reliable,
        )
        qos_turn_context = self._qos_profile(
            depth_parameter="turn_context.qos.depth",
            reliable_parameter="turn_context.qos.reliable",
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
        self.control_sub = self.create_subscription(
            VadState,
            self.control_config.topic,
            self._make_vad_control_callback(self.control_config),
            qos_control,
        )
        self.turn_context_sub = self.create_subscription(
            TurnContext,
            self.turn_context_topic,
            self.on_turn_context,
            qos_turn_context,
        )

        self.get_logger().info(
            "fa_turn_detector started: "
            f"audio={self.audio_topic} "
            f"control={self.control_config.control_id}:{self.control_config.topic} "
            f"turn_context={self.turn_context_topic} "
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
        self.declare_parameter("output_topic", Parameter.Type.STRING)
        self.declare_parameter("expected_source_id", Parameter.Type.STRING)
        self.declare_parameter("control.default_enabled", Parameter.Type.BOOL)
        self.declare_parameter("control.inputs", Parameter.Type.STRING_ARRAY)
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
        self.declare_parameter("output.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("output.qos.reliable", Parameter.Type.BOOL)

    def _validate_identity_contract(self) -> None:
        for topic_name, topic_value in (
            ("audio_topic", self.audio_topic),
            (f"control.{self.control_config.control_id}.topic", self.control_config.topic),
            ("turn_context_topic", self.turn_context_topic),
            ("output_topic", self.output_topic),
        ):
            if self._same_identity_string(self.expected_stream_id, topic_value):
                raise RuntimeError(
                    f"expected_stream_id must be distinct from ROS {topic_name}"
                )
        if self._same_identity_string(self.audio_topic, self.output_topic):
            raise RuntimeError("audio_topic must be distinct from output_topic")

    def _validate_control_default_enabled(self) -> None:
        if self.control_default_enabled:
            raise RuntimeError(
                "control.default_enabled must be false for fa_turn_detector; "
                "turn-end detection requires explicit control events"
            )

    def _validate_control_binding_contract(self) -> None:
        if self.control_config.source_id != self.expected_source_id:
            raise RuntimeError(
                f"control.{self.control_config.control_id}.source_id must match expected_source_id"
            )
        if self.control_config.stream_id != self.expected_stream_id:
            raise RuntimeError(
                f"control.{self.control_config.control_id}.stream_id must match expected_stream_id"
            )

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

    def _load_control_config(self) -> ControlInputConfig:
        control_ids = FaTurnDetectorNode._string_tuple_parameter(self, "control.inputs")
        if len(control_ids) != len(set(control_ids)):
            raise RuntimeError("control.inputs must not contain duplicate IDs")
        if len(control_ids) != 1:
            raise RuntimeError("control.inputs must contain exactly one ID")
        control_id = control_ids[0]
        normalized_id = control_id.strip()
        if not normalized_id:
            raise RuntimeError("control.inputs must not contain empty IDs")
        if normalized_id != control_id:
            raise RuntimeError("control.inputs IDs must not contain surrounding whitespace")
        prefix = f"control.{normalized_id}"
        FaTurnDetectorNode._declare_control_parameters(self, prefix)
        return ControlInputConfig(
            control_id=normalized_id,
            action=FaTurnDetectorNode._control_action_parameter(self, f"{prefix}.action"),
            topic=FaTurnDetectorNode._required_string_parameter(self, f"{prefix}.topic"),
            msg_type=FaTurnDetectorNode._control_msg_type_parameter(
                self,
                f"{prefix}.msg_type",
            ),
            source_id=FaTurnDetectorNode._required_string_parameter(
                self,
                f"{prefix}.source_id",
            ),
            stream_id=FaTurnDetectorNode._required_string_parameter(
                self,
                f"{prefix}.stream_id",
            ),
            active_field=FaTurnDetectorNode._vad_control_field_parameter(
                self,
                f"{prefix}.active_field",
                expected_field="is_speech",
            ),
            start_field=FaTurnDetectorNode._vad_control_field_parameter(
                self,
                f"{prefix}.start_field",
                expected_field="start",
            ),
            end_field=FaTurnDetectorNode._vad_control_field_parameter(
                self,
                f"{prefix}.end_field",
                expected_field="end",
            ),
            close_on=FaTurnDetectorNode._control_close_policy_parameter(
                self,
                f"{prefix}.close_on",
            ),
            qos_depth=FaTurnDetectorNode._positive_integer_parameter(
                self,
                f"{prefix}.qos.depth",
            ),
            qos_reliable=FaTurnDetectorNode._bool_parameter(
                self,
                f"{prefix}.qos.reliable",
            ),
        )

    def _declare_control_parameters(self, prefix: str) -> None:
        self.declare_parameter(f"{prefix}.action", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.topic", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.msg_type", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.source_id", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.stream_id", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.active_field", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.start_field", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.end_field", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.close_on", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter(f"{prefix}.qos.reliable", Parameter.Type.BOOL)

    def _string_tuple_parameter(self, name: str) -> tuple[str, ...]:
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.STRING_ARRAY:
            raise RuntimeError(f"{name} must be a string array")
        return tuple(parameter.get_parameter_value().string_array_value)

    def _control_action_parameter(self, name: str) -> str:
        value = FaTurnDetectorNode._string_parameter(self, name).strip()
        if value != "topic":
            raise RuntimeError(f"{name} must be topic")
        return value

    def _control_msg_type_parameter(self, name: str) -> str:
        value = FaTurnDetectorNode._string_parameter(self, name).strip()
        if value != "fa_interfaces/msg/VadState":
            raise RuntimeError(f"{name} must be fa_interfaces/msg/VadState")
        return value

    def _vad_control_field_parameter(self, name: str, *, expected_field: str) -> str:
        value = FaTurnDetectorNode._string_parameter(self, name).strip()
        if value != expected_field:
            raise RuntimeError(f"{name} must be {expected_field}")
        return value

    def _control_close_policy_parameter(self, name: str) -> str:
        value = FaTurnDetectorNode._string_parameter(self, name).strip()
        if value != "end_or_active_falling":
            raise RuntimeError(f"{name} must be end_or_active_falling")
        return value

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
            raise RuntimeError("control qos depth must be greater than zero")
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
            self.is_speech = False
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
            self.is_speech = False

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

    def _make_vad_control_callback(
        self,
        config: ControlInputConfig,
    ) -> Callable[[VadState], None]:
        def callback(msg: VadState) -> None:
            self.on_control_event(self._vad_state_to_control_event(config, msg))

        return callback

    @staticmethod
    def _vad_state_to_control_event(
        config: ControlInputConfig,
        msg: VadState,
    ) -> ControlEvent:
        return ControlEvent(
            control_id=config.control_id,
            source_id=msg.source_id,
            stream_id=msg.stream_id,
            active=bool(msg.is_speech),
            start=bool(msg.start),
            end=bool(msg.end),
        )

    def on_control_event(self, event: ControlEvent) -> None:
        try:
            self._validate_control_event_identity(event, config=self.control_config)
        except ValueError as exc:
            self.get_logger().error(f"Dropping control event {event.control_id}: {exc}")
            return
        if not self._context_active:
            self.is_speech = event.active
            return
        prev_is_speech = self.is_speech
        self.is_speech = event.active
        if event.end or (prev_is_speech and not self.is_speech):
            self._detect_turn_end()

    @staticmethod
    def _validate_control_event_identity(
        event: ControlEvent,
        *,
        config: ControlInputConfig,
    ) -> None:
        if not event.source_id or not event.stream_id:
            raise ValueError("source_id and stream_id are required")
        if event.source_id != config.source_id or event.stream_id != config.stream_id:
            raise ValueError("source_id/stream_id mismatch")

    def _detect_turn_end(self) -> None:
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
        out.probability = float(result.probability)
        out.is_end = bool(result.is_end)
        self.turn_end_pub.publish(out)

        self.get_logger().info(
            f"Turn end probability: {result.probability:.3f} is_end={str(out.is_end).lower()}"
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
