#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Float32

from fa_interfaces.msg import AudioFrame, VadState
from fa_vad_py.silero import SileroVAD


def _convert_to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1 or samples.ndim == 1:
        return samples.astype(np.float32)
    reshaped = samples.reshape(-1, channels)
    return reshaped.mean(axis=1).astype(np.float32)


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32)
    ratio = dst_rate / float(src_rate)
    dst_len = max(1, int(math.floor(samples.size * ratio)))
    positions = np.linspace(0, samples.size, dst_len, endpoint=False)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


def _float_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


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

        self.declare_parameter("silero.repo_dir", "")
        self.declare_parameter("silero.allow_online", False)

        self.declare_parameter("log_speech_events", True)

        self.declare_parameter("qos.depth", 10)
        self.declare_parameter("qos.reliable", False)

        self._input_topic = str(self.get_parameter("input_topic").value)
        self._output_topic = str(self.get_parameter("output_topic").value)
        self._vad_state_topic = str(self.get_parameter("vad_state_topic").value)
        self._probability_topic = str(self.get_parameter("probability_topic").value)

        self._publish_vad_state = bool(self.get_parameter("publish_vad_state").value)
        self._publish_probability = bool(self.get_parameter("publish_probability").value)
        self._log_events = bool(self.get_parameter("log_speech_events").value)

        self._target_sample_rate = int(self.get_parameter("target_sample_rate").value)
        threshold_start = float(self.get_parameter("threshold_start").value)
        threshold_end = float(self.get_parameter("threshold_end").value)
        hangover_ms = int(self.get_parameter("hangover_ms").value)

        silero_repo_dir = str(self.get_parameter("silero.repo_dir").value).strip()
        allow_online = bool(self.get_parameter("silero.allow_online").value)

        depth = int(self.get_parameter("qos.depth").value)
        reliable = bool(self.get_parameter("qos.reliable").value)

        qos = QoSProfile(depth=max(1, depth))
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

        self._vad = SileroVAD(
            sample_rate=self._target_sample_rate,
            frame_ms=20,
            hangover_ms=hangover_ms,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            silero_repo_dir=silero_repo_dir or None,
            allow_online=allow_online,
        )

        self._last_is_speech: Optional[bool] = None

        self.get_logger().info(
            "FA VAD (Silero): input=%s output=%s vad_state=%s "
            "target_sr=%d start=%.2f end=%.2f hangover=%dms online=%s repo_dir=%s",
            self._input_topic,
            self._output_topic,
            self._vad_state_topic if self._publish_vad_state else "(disabled)",
            self._target_sample_rate,
            threshold_start,
            threshold_end,
            hangover_ms,
            str(allow_online).lower(),
            silero_repo_dir if silero_repo_dir else "(default)",
        )

    def _on_audio_frame(self, msg: AudioFrame) -> None:
        samples = self._audio_to_float(msg)
        if samples.size == 0:
            return

        resampled = _resample_linear(samples, int(msg.sample_rate), self._target_sample_rate)
        pcm_bytes = _float_to_pcm16(resampled)
        probability, is_speech, start, end = self._vad.update(pcm_bytes)

        if self._log_events:
            if start:
                self.get_logger().info("Speech START (prob=%.2f)", probability)
            if end:
                self.get_logger().info("Speech END (prob=%.2f)", probability)

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

    @staticmethod
    def _audio_to_float(msg: AudioFrame) -> np.ndarray:
        if not msg.data:
            return np.zeros(0, dtype=np.float32)

        if msg.bit_depth == 16:
            samples = np.frombuffer(bytes(msg.data), dtype=np.int16).astype(np.float32) / 32768.0
        elif msg.bit_depth == 32:
            samples = np.frombuffer(bytes(msg.data), dtype=np.float32)
        else:
            return np.zeros(0, dtype=np.float32)

        if msg.channels and msg.channels > 1:
            samples = _convert_to_mono(samples, int(msg.channels))

        return samples


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

