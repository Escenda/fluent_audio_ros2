#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

import numpy as np
import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AudioFrame, MfccFrame
from fa_mfcc_py.backends.mfcc import (
    InternalMfccBackend,
    MfccConfig,
)


class FaMfccNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_mfcc")
        self._load_parameters()

        self.backend = InternalMfccBackend(
            MfccConfig(
                sample_rate=self.expected_sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                n_mfcc=self.n_mfcc,
                f_min_hz=self.f_min_hz,
                f_max_hz=self.f_max_hz,
                log_floor=self.log_floor,
                dct_type=self.dct_type,
                normalization=self.normalization,
            )
        )

        qos = QoSProfile(depth=self.qos_depth)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (
            ReliabilityPolicy.RELIABLE if self.qos_reliable else ReliabilityPolicy.BEST_EFFORT
        )

        self.feature_pub = self.create_publisher(MfccFrame, self.output_topic, qos)
        self.audio_sub = self.create_subscription(
            AudioFrame,
            self.input_topic,
            self.on_audio,
            qos,
        )

        self.get_logger().info(
            "fa_mfcc started: "
            f"input={self.input_topic} output={self.output_topic} "
            f"n_fft={self.n_fft} hop={self.hop_length} "
            f"n_mels={self.n_mels} n_mfcc={self.n_mfcc}"
        )

    def _load_parameters(self) -> None:
        self.input_topic = self._required_string("input_topic")
        self.output_topic = self._required_string("output_topic")
        self.backend_name = self._required_string("backend.name")
        self.expected_sample_rate = self._required_integer("expected.sample_rate")
        self.expected_channels = self._required_integer("expected.channels")
        self.expected_encoding = self._required_string("expected.encoding")
        self.expected_bit_depth = self._required_integer("expected.bit_depth")
        self.expected_layout = self._required_string("expected.layout")
        self.n_fft = self._required_integer("feature.n_fft")
        self.hop_length = self._required_integer("feature.hop_length")
        self.n_mels = self._required_integer("feature.n_mels")
        self.n_mfcc = self._required_integer("feature.n_mfcc")
        self.f_min_hz = self._required_double("feature.f_min_hz")
        self.f_max_hz = self._required_double("feature.f_max_hz")
        self.log_floor = self._required_double("feature.log_floor")
        self.dct_type = self._required_string("feature.dct_type")
        self.normalization = self._required_string("feature.normalization")
        self.qos_depth = self._required_integer("qos.depth")
        self.qos_reliable = self._required_bool("qos.reliable")

        if not self.input_topic:
            raise RuntimeError("input_topic is required")
        if not self.output_topic:
            raise RuntimeError("output_topic is required")
        if self.backend_name != InternalMfccBackend.name:
            raise RuntimeError("backend.name must be internal_mfcc")
        if self.expected_sample_rate <= 0:
            raise RuntimeError("expected.sample_rate must be > 0")
        if self.expected_channels != 1:
            raise RuntimeError("fa_mfcc currently requires expected.channels=1")
        if self.expected_encoding != "FLOAT32LE":
            raise RuntimeError("fa_mfcc requires expected.encoding=FLOAT32LE")
        if self.expected_bit_depth != 32:
            raise RuntimeError("fa_mfcc requires expected.bit_depth=32")
        if self.expected_layout != "interleaved":
            raise RuntimeError("fa_mfcc requires expected.layout=interleaved")
        if self.qos_depth <= 0:
            raise RuntimeError("qos.depth must be > 0")

    def _required_string(self, name: str) -> str:
        self.declare_parameter(name, Parameter.Type.STRING)
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.STRING:
            raise RuntimeError(f"{name} must be a string parameter")
        value = parameter.get_parameter_value().string_value.strip()
        if not value:
            raise RuntimeError(f"{name} is required")
        return value

    def _required_integer(self, name: str) -> int:
        self.declare_parameter(name, Parameter.Type.INTEGER)
        try:
            return int(self.get_parameter(name).get_parameter_value().integer_value)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc

    def _required_double(self, name: str) -> float:
        self.declare_parameter(name, Parameter.Type.DOUBLE)
        try:
            return float(self.get_parameter(name).get_parameter_value().double_value)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc

    def _required_bool(self, name: str) -> bool:
        self.declare_parameter(name, Parameter.Type.BOOL)
        try:
            return bool(self.get_parameter(name).get_parameter_value().bool_value)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc

    def on_audio(self, msg: AudioFrame) -> None:
        try:
            samples = self._frame_to_float(msg)
            result = self.backend.compute(samples)
        except ValueError as exc:
            self.get_logger().warning(f"Dropping AudioFrame before MFCC publish: {exc}")
            return
        except RuntimeError as exc:
            self.get_logger().error(f"fa_mfcc backend failure: {exc}")
            raise

        out = MfccFrame()
        out.header = msg.header
        out.source_id = msg.source_id
        out.stream_id = self.output_topic
        out.sample_rate = int(msg.sample_rate)
        out.input_sample_count = int(samples.size)
        out.n_fft = int(self.n_fft)
        out.hop_length = int(self.hop_length)
        out.n_mels = int(self.n_mels)
        out.n_mfcc = int(self.n_mfcc)
        out.frame_count = int(result.frame_count)
        out.f_min_hz = float(self.f_min_hz)
        out.f_max_hz = float(self.f_max_hz)
        out.log_floor = float(self.log_floor)
        out.dct_type = self.dct_type
        out.normalization = self.normalization
        out.layout = "frames_by_coefficients"
        out.values = result.values.reshape(-1).astype(np.float32).tolist()
        self.feature_pub.publish(out)

    def _frame_to_float(self, msg: AudioFrame) -> np.ndarray:
        if not msg.data:
            raise ValueError("AudioFrame data is required")
        if not msg.source_id or not msg.stream_id:
            raise ValueError("AudioFrame source_id and stream_id are required")
        if msg.stream_id != self.input_topic:
            raise ValueError("AudioFrame stream_id must match input_topic")
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
    node = FaMfccNode()
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
