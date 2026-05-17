#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AudioFrame, TurnContext, TurnEnd, VadState
from fa_turn_detector_py.backends.base import TurnDetectorBackend
from fa_turn_detector_py.backends.smart_turn_onnx import SmartTurnOnnxBackend


class FaTurnDetectorNode(Node):
    """Turn detection node using a local Smart Turn v3 ONNX model."""

    def __init__(self) -> None:
        super().__init__("fa_turn_detector")

        self.declare_parameter("audio_topic", "audio/frame")
        self.declare_parameter("vad_topic", "voice/vad_state")
        self.declare_parameter("turn_context_topic", "conversation/turn_context")
        self.declare_parameter("output_topic", "voice/turn_end")
        self.declare_parameter("backend.name", "")
        self.declare_parameter("backend.model_path", "")
        self.declare_parameter("backend.threshold", 0.5)
        self.declare_parameter("backend.execution_provider", "")
        self.declare_parameter("backend.command", "")
        self.declare_parameter("backend.args", [])
        self.declare_parameter("backend.health_args", [])
        self.declare_parameter("backend.timeout_sec", 5.0)
        self.declare_parameter("backend.workspace_dir", "/tmp/fluent_audio_fa_turn_detector")
        self.declare_parameter("backend.cleanup_audio_files", True)

        self.audio_topic = str(self.get_parameter("audio_topic").value)
        self.vad_topic = str(self.get_parameter("vad_topic").value)
        self.turn_context_topic = str(self.get_parameter("turn_context_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        self.backend = self._load_backend()
        self.audio_buffer: deque[float] = deque(maxlen=self.backend.sample_rate * 10)
        self.is_speech = False
        self._active_session_id = ""
        self._active_user_turn_id = 0
        self._context_active = False

        qos_sensor = QoSProfile(depth=10)
        qos_sensor.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_sensor.history = HistoryPolicy.KEEP_LAST

        qos_reliable = QoSProfile(depth=10)
        qos_reliable.reliability = ReliabilityPolicy.RELIABLE
        qos_reliable.history = HistoryPolicy.KEEP_LAST

        self.turn_end_pub = self.create_publisher(TurnEnd, self.output_topic, qos_reliable)
        self.audio_sub = self.create_subscription(
            AudioFrame,
            self.audio_topic,
            self.on_audio,
            qos_sensor,
        )
        self.vad_sub = self.create_subscription(
            VadState,
            self.vad_topic,
            self.on_vad,
            qos_sensor,
        )
        self.turn_context_sub = self.create_subscription(
            TurnContext,
            self.turn_context_topic,
            self.on_turn_context,
            qos_reliable,
        )

        self.get_logger().info(
            "fa_turn_detector started: audio=%s vad=%s turn_context=%s output=%s backend.name=%s model=%s",
            self.audio_topic,
            self.vad_topic,
            self.turn_context_topic,
            self.output_topic,
            self.backend.name,
            self.backend.model_path,
        )

    def _load_backend(self) -> TurnDetectorBackend:
        backend_name = str(self.get_parameter("backend.name").value).strip()
        if not backend_name:
            raise RuntimeError("backend.name is required")
        if backend_name != SmartTurnOnnxBackend.name:
            raise RuntimeError(f"unsupported turn detector backend.name: {backend_name}")
        return SmartTurnOnnxBackend(
            model_path=str(self.get_parameter("backend.model_path").value).strip(),
            threshold=float(self.get_parameter("backend.threshold").value),
            execution_provider=str(self.get_parameter("backend.execution_provider").value).strip(),
            command=str(self.get_parameter("backend.command").value).strip(),
            args=self._string_tuple_parameter("backend.args"),
            health_args=self._string_tuple_parameter("backend.health_args"),
            timeout_sec=float(self.get_parameter("backend.timeout_sec").value),
            workspace_dir=str(self.get_parameter("backend.workspace_dir").value).strip(),
            cleanup_audio_files=bool(self.get_parameter("backend.cleanup_audio_files").value),
        )

    def _string_tuple_parameter(self, name: str) -> tuple[str, ...]:
        value = self.get_parameter(name).value
        if not isinstance(value, (list, tuple)):
            raise RuntimeError(f"{name} must be a string list")
        return tuple(str(item) for item in value)

    def on_turn_context(self, msg: TurnContext) -> None:
        self._active_session_id = str(msg.session_id)
        self._active_user_turn_id = int(msg.user_turn_id)
        self._context_active = bool(msg.active) and bool(msg.session_id)
        if not self._context_active:
            self.audio_buffer.clear()

    def on_audio(self, msg: AudioFrame) -> None:
        if not self._context_active:
            return
        try:
            if int(msg.sample_rate) != self.backend.sample_rate:
                raise ValueError(
                    "AudioFrame sample_rate must match backend sample_rate "
                    f"{self.backend.sample_rate}, got {msg.sample_rate}"
                )
            audio_data = self._frame_to_float(msg)
        except ValueError as exc:
            self.get_logger().error("Dropping invalid AudioFrame: %s", exc)
            return
        self.audio_buffer.extend(audio_data.tolist())

    def on_vad(self, msg: VadState) -> None:
        if not self._context_active:
            self.is_speech = bool(msg.is_speech)
            return
        prev_is_speech = self.is_speech
        self.is_speech = bool(msg.is_speech)
        if msg.end or (prev_is_speech and not self.is_speech):
            self._detect_turn_end()

    def _detect_turn_end(self) -> None:
        if len(self.audio_buffer) < self.backend.min_samples:
            self.get_logger().debug(
                "Not enough audio data for turn detection: %d samples",
                len(self.audio_buffer),
            )
            return

        available_samples = min(len(self.audio_buffer), self.backend.max_samples)
        audio = np.asarray(list(self.audio_buffer)[-available_samples:], dtype=np.float32)
        try:
            result = self.backend.detect(audio)
        except Exception as exc:
            self.get_logger().fatal("Turn detection backend failed: %s", exc)
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
            "Turn end probability: %.3f is_end=%s",
            result.probability,
            str(out.is_end).lower(),
        )

    @staticmethod
    def _frame_to_float(msg: AudioFrame) -> np.ndarray:
        if not msg.data:
            raise ValueError("AudioFrame data is required")
        if not msg.source_id or not msg.stream_id:
            raise ValueError("AudioFrame source_id and stream_id are required")
        if msg.layout != "interleaved":
            raise ValueError(f"AudioFrame layout must be interleaved, got {msg.layout}")
        if int(msg.channels) != 1:
            raise ValueError(f"AudioFrame channels must be 1, got {msg.channels}")
        if msg.encoding != "FLOAT32LE":
            raise ValueError(f"AudioFrame encoding must be FLOAT32LE, got {msg.encoding}")
        if int(msg.bit_depth) != 32:
            raise ValueError(f"AudioFrame bit_depth must be 32, got {msg.bit_depth}")
        if len(msg.data) % np.dtype(np.float32).itemsize != 0:
            raise ValueError("AudioFrame float32 data length is not byte-aligned")
        samples = np.frombuffer(bytes(msg.data), dtype=np.float32)
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
