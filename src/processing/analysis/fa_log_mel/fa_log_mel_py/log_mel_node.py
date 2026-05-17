#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AudioFrame, LogMelFrame
from fa_log_mel_py.backends.log_mel import (
    InternalLogMelBackend,
    LogMelConfig,
)


class FaLogMelNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_log_mel")
        self._load_parameters()

        self.backend = InternalLogMelBackend(
            LogMelConfig(
                sample_rate=self.expected_sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min_hz=self.f_min_hz,
                f_max_hz=self.f_max_hz,
                log_floor=self.log_floor,
            )
        )

        qos = QoSProfile(depth=self.qos_depth)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (
            ReliabilityPolicy.RELIABLE if self.qos_reliable else ReliabilityPolicy.BEST_EFFORT
        )

        self.feature_pub = self.create_publisher(LogMelFrame, self.output_topic, qos)
        self.audio_sub = self.create_subscription(
            AudioFrame,
            self.input_topic,
            self.on_audio,
            qos,
        )

        self.get_logger().info(
            "fa_log_mel started: input=%s output=%s n_fft=%d hop=%d n_mels=%d",
            self.input_topic,
            self.output_topic,
            self.n_fft,
            self.hop_length,
            self.n_mels,
        )

    def _load_parameters(self) -> None:
        self.declare_parameter("input_topic", "audio/features/input")
        self.declare_parameter("output_topic", "audio/features/log_mel")
        self.declare_parameter("backend.name", "")
        self.declare_parameter("expected.sample_rate", 16000)
        self.declare_parameter("expected.channels", 1)
        self.declare_parameter("expected.encoding", "FLOAT32LE")
        self.declare_parameter("expected.bit_depth", 32)
        self.declare_parameter("expected.layout", "interleaved")
        self.declare_parameter("feature.n_fft", 400)
        self.declare_parameter("feature.hop_length", 160)
        self.declare_parameter("feature.n_mels", 80)
        self.declare_parameter("feature.f_min_hz", 0.0)
        self.declare_parameter("feature.f_max_hz", 8000.0)
        self.declare_parameter("feature.log_floor", 1.0e-10)
        self.declare_parameter("qos.depth", 10)
        self.declare_parameter("qos.reliable", False)

        self.input_topic = str(self.get_parameter("input_topic").value).strip()
        self.output_topic = str(self.get_parameter("output_topic").value).strip()
        self.backend_name = str(self.get_parameter("backend.name").value).strip()
        self.expected_sample_rate = int(self.get_parameter("expected.sample_rate").value)
        self.expected_channels = int(self.get_parameter("expected.channels").value)
        self.expected_encoding = str(self.get_parameter("expected.encoding").value).strip()
        self.expected_bit_depth = int(self.get_parameter("expected.bit_depth").value)
        self.expected_layout = str(self.get_parameter("expected.layout").value).strip()
        self.n_fft = int(self.get_parameter("feature.n_fft").value)
        self.hop_length = int(self.get_parameter("feature.hop_length").value)
        self.n_mels = int(self.get_parameter("feature.n_mels").value)
        self.f_min_hz = float(self.get_parameter("feature.f_min_hz").value)
        self.f_max_hz = float(self.get_parameter("feature.f_max_hz").value)
        self.log_floor = float(self.get_parameter("feature.log_floor").value)
        self.qos_depth = int(self.get_parameter("qos.depth").value)
        self.qos_reliable = bool(self.get_parameter("qos.reliable").value)

        if not self.input_topic:
            raise RuntimeError("input_topic is required")
        if not self.output_topic:
            raise RuntimeError("output_topic is required")
        if self.backend_name != InternalLogMelBackend.name:
            raise RuntimeError("backend.name must be internal_log_mel")
        if self.expected_sample_rate <= 0:
            raise RuntimeError("expected.sample_rate must be > 0")
        if self.expected_channels != 1:
            raise RuntimeError("fa_log_mel currently requires expected.channels=1")
        if self.expected_encoding != "FLOAT32LE":
            raise RuntimeError("fa_log_mel requires expected.encoding=FLOAT32LE")
        if self.expected_bit_depth != 32:
            raise RuntimeError("fa_log_mel requires expected.bit_depth=32")
        if self.expected_layout != "interleaved":
            raise RuntimeError("fa_log_mel requires expected.layout=interleaved")
        if self.qos_depth <= 0:
            raise RuntimeError("qos.depth must be > 0")

    def on_audio(self, msg: AudioFrame) -> None:
        try:
            samples = self._frame_to_float(msg)
            result = self.backend.compute(samples)
        except (RuntimeError, ValueError) as exc:
            self.get_logger().warning("Dropping AudioFrame before log-mel publish: %s", exc)
            return

        out = LogMelFrame()
        out.header = msg.header
        out.source_id = msg.source_id
        out.stream_id = self.output_topic
        out.sample_rate = int(msg.sample_rate)
        out.input_sample_count = int(samples.size)
        out.n_fft = int(self.n_fft)
        out.hop_length = int(self.hop_length)
        out.n_mels = int(self.n_mels)
        out.frame_count = int(result.frame_count)
        out.f_min_hz = float(self.f_min_hz)
        out.f_max_hz = float(self.f_max_hz)
        out.log_floor = float(self.log_floor)
        out.layout = "frames_by_mels"
        out.values = result.values.reshape(-1).astype(np.float32).tolist()
        self.feature_pub.publish(out)

    def _frame_to_float(self, msg: AudioFrame) -> np.ndarray:
        if not msg.data:
            raise ValueError("AudioFrame data is required")
        if not msg.source_id or not msg.stream_id:
            raise ValueError("AudioFrame source_id and stream_id are required")
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
    node = FaLogMelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
