#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AudioEmbeddingFrame, AudioFrame
from fa_audio_embedding_py.backends.base import AudioEmbeddingBackend, AudioEmbeddingRequest
from fa_audio_embedding_py.backends.factory import (
    AudioEmbeddingBackendSettings,
    build_audio_embedding_backend,
)


class FaAudioEmbeddingNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_audio_embedding")
        self._declare_parameters()
        self._load_parameters()
        self.backend = self._load_backend()

        qos = QoSProfile(depth=self.qos_depth)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (
            ReliabilityPolicy.RELIABLE if self.qos_reliable else ReliabilityPolicy.BEST_EFFORT
        )

        self.embedding_pub = self.create_publisher(AudioEmbeddingFrame, self.output_topic, qos)
        self.audio_sub = self.create_subscription(AudioFrame, self.input_topic, self.on_audio, qos)
        self.get_logger().info(
            "fa_audio_embedding started: input=%s output=%s source=%s stream=%s dimension=%d backend.name=%s",
            self.input_topic,
            self.output_topic,
            self.expected_source_id,
            self.expected_stream_id,
            self.embedding_dimension,
            self.backend.name,
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_topic", "audio/embedding/input")
        self.declare_parameter("output_topic", "audio/embedding/frame")
        self.declare_parameter("expected_source_id", "")
        self.declare_parameter("expected_stream_id", "")
        self.declare_parameter("expected.sample_rate", 16000)
        self.declare_parameter("expected.channels", 1)
        self.declare_parameter("expected.encoding", "FLOAT32LE")
        self.declare_parameter("expected.bit_depth", 32)
        self.declare_parameter("expected.layout", "interleaved")
        self.declare_parameter("embedding.dimension", 0)
        self.declare_parameter("backend.name", "")
        self.declare_parameter("backend.command", "")
        self.declare_parameter("backend.model_id", "")
        self.declare_parameter("backend.model_path", "")
        self.declare_parameter("backend.args", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("backend.payload_encoding", "float32le_raw")
        self.declare_parameter("backend.timeout_sec", 30.0)
        self.declare_parameter("backend.workspace_dir", "/tmp/fa_audio_embedding")
        self.declare_parameter("backend.cleanup_audio_files", True)
        self.declare_parameter("qos.depth", 10)
        self.declare_parameter("qos.reliable", False)

    def _load_parameters(self) -> None:
        self.input_topic = self._string_parameter("input_topic").strip()
        self.output_topic = self._string_parameter("output_topic").strip()
        self.expected_source_id = self._string_parameter("expected_source_id").strip()
        self.expected_stream_id = self._string_parameter("expected_stream_id").strip()
        self.expected_sample_rate = self._integer_parameter("expected.sample_rate")
        self.expected_channels = self._integer_parameter("expected.channels")
        self.expected_encoding = self._string_parameter("expected.encoding").strip()
        self.expected_bit_depth = self._integer_parameter("expected.bit_depth")
        self.expected_layout = self._string_parameter("expected.layout").strip()
        self.embedding_dimension = self._integer_parameter("embedding.dimension")
        self.workspace_dir = Path(self._string_parameter("backend.workspace_dir")).expanduser()
        self.cleanup_audio_files = self._bool_parameter("backend.cleanup_audio_files")
        self.qos_depth = self._integer_parameter("qos.depth")
        self.qos_reliable = self._bool_parameter("qos.reliable")

        if not self.input_topic:
            raise RuntimeError("input_topic is required")
        if not self.output_topic:
            raise RuntimeError("output_topic is required")
        if not self.expected_source_id:
            raise RuntimeError("expected_source_id is required")
        if not self.expected_stream_id:
            raise RuntimeError("expected_stream_id is required")
        if self.expected_sample_rate <= 0:
            raise RuntimeError("expected.sample_rate must be > 0")
        if self.expected_channels != 1:
            raise RuntimeError("fa_audio_embedding currently requires expected.channels=1")
        if self.expected_encoding != "FLOAT32LE":
            raise RuntimeError("fa_audio_embedding requires expected.encoding=FLOAT32LE")
        if self.expected_bit_depth != 32:
            raise RuntimeError("fa_audio_embedding requires expected.bit_depth=32")
        if self.expected_layout != "interleaved":
            raise RuntimeError("fa_audio_embedding requires expected.layout=interleaved")
        if self.embedding_dimension <= 0:
            raise RuntimeError("embedding.dimension must be > 0")
        if self.qos_depth <= 0:
            raise RuntimeError("qos.depth must be > 0")

    def _load_backend(self) -> AudioEmbeddingBackend:
        return build_audio_embedding_backend(
            AudioEmbeddingBackendSettings(
                name=self._string_parameter("backend.name"),
                command=self._string_parameter("backend.command"),
                model_id=self._string_parameter("backend.model_id"),
                model_path=self._string_parameter("backend.model_path"),
                args=self._string_array_parameter("backend.args"),
                payload_encoding=self._string_parameter("backend.payload_encoding"),
                timeout_sec=self._double_parameter("backend.timeout_sec"),
                workspace_dir=self.workspace_dir,
                cleanup_audio_files=self.cleanup_audio_files,
                dimension=self.embedding_dimension,
            )
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
        return tuple(str(item) for item in parameter.value)

    def on_audio(self, msg: AudioFrame) -> None:
        try:
            samples = self._frame_to_float(msg)
        except ValueError as exc:
            self.get_logger().warning(f"Dropping AudioFrame before embedding publish: {exc}")
            return

        try:
            result = self.backend.embed(
                AudioEmbeddingRequest(
                    samples=samples,
                    sample_rate=int(msg.sample_rate),
                    source_id=msg.source_id,
                    stream_id=msg.stream_id,
                )
            )
            if result.embedding.size != self.embedding_dimension:
                raise RuntimeError(
                    "audio embedding dimension mismatch: "
                    f"expected {self.embedding_dimension}, got {result.embedding.size}"
                )
        except TimeoutError as exc:
            self.get_logger().error(f"fa_audio_embedding backend timeout: {exc}")
            raise
        except (ValueError, RuntimeError) as exc:
            self.get_logger().error(f"fa_audio_embedding backend failure: {exc}")
            raise

        out = AudioEmbeddingFrame()
        out.header = msg.header
        out.source_id = msg.source_id
        out.stream_id = msg.stream_id
        out.model_id = result.model_id
        out.sample_rate = int(msg.sample_rate)
        out.input_sample_count = int(samples.size)
        out.dimension = int(result.embedding.size)
        out.payload_encoding = "float32"
        out.embedding = [float(value) for value in result.embedding]
        self.embedding_pub.publish(out)

    def _frame_to_float(self, msg: AudioFrame) -> np.ndarray:
        if not msg.data:
            raise ValueError("AudioFrame data is required")
        if not msg.source_id or not msg.stream_id:
            raise ValueError("AudioFrame source_id and stream_id are required")
        if msg.source_id != self.expected_source_id:
            raise ValueError("AudioFrame source_id must match expected_source_id")
        if msg.stream_id != self.expected_stream_id:
            raise ValueError("AudioFrame stream_id must match expected_stream_id")
        if msg.layout != self.expected_layout:
            raise ValueError(f"AudioFrame layout must be {self.expected_layout}")
        if msg.encoding != self.expected_encoding:
            raise ValueError(f"AudioFrame encoding must be {self.expected_encoding}")
        if int(msg.bit_depth) != self.expected_bit_depth:
            raise ValueError(f"AudioFrame bit_depth must be {self.expected_bit_depth}")
        if int(msg.sample_rate) != self.expected_sample_rate:
            raise ValueError(
                "AudioFrame sample_rate must match expected.sample_rate "
                f"{self.expected_sample_rate}"
            )
        if int(msg.channels) != self.expected_channels:
            raise ValueError(f"AudioFrame channels must be {self.expected_channels}")
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
    node = FaAudioEmbeddingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
