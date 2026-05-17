#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Float32

from fa_interfaces.msg import AudioFrame, VadState
from fa_vad_py.backends.base import VADBackend
from fa_vad_py.backends.silero import SileroVAD
from fa_vad_py.contracts import (
    audio_frame_to_float_samples,
    validate_node_config,
    validate_qos_depth,
)


def _float_to_pcm16(samples: np.ndarray) -> bytes:
    return (samples * 32767.0).astype(np.int16).tobytes()


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
            [
                "--audio",
                "{audio}",
                "--model",
                "{model}",
                "--provider",
                "{provider}",
                "--sample-rate",
                "{sample_rate}",
            ],
        )
        self.declare_parameter("backend.timeout_sec", 1.0)
        self.declare_parameter("backend.workspace_dir", "/tmp/fluent_audio/fa_vad")
        self.declare_parameter("backend.cleanup_audio_files", True)

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
        validate_node_config(
            target_sample_rate=self._target_sample_rate,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            hangover_ms=hangover_ms,
        )

        model_path = str(self.get_parameter("backend.model_path").value).strip()
        execution_provider = str(
            self.get_parameter("backend.execution_provider").value
        ).strip()
        command = str(self.get_parameter("backend.command").value).strip()
        backend_args = tuple(
            str(item) for item in self.get_parameter("backend.args").value
        )
        timeout_sec = float(self.get_parameter("backend.timeout_sec").value)
        workspace_dir = str(self.get_parameter("backend.workspace_dir").value).strip()
        cleanup_audio_files = bool(
            self.get_parameter("backend.cleanup_audio_files").value
        )

        depth = validate_qos_depth(int(self.get_parameter("qos.depth").value))
        reliable = bool(self.get_parameter("qos.reliable").value)

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
            "FA VAD (Silero): input=%s output=%s vad_state=%s "
            "target_sr=%d start=%.2f end=%.2f hangover=%dms provider=%s model_path=%s command=%s",
            self._input_topic,
            self._output_topic,
            self._vad_state_topic if self._publish_vad_state else "(disabled)",
            self._target_sample_rate,
            threshold_start,
            threshold_end,
            hangover_ms,
            execution_provider,
            model_path,
            command,
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
        backend_name = str(self.get_parameter("backend.name").value).strip()
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

    def _on_audio_frame(self, msg: AudioFrame) -> None:
        try:
            if int(msg.sample_rate) != self._target_sample_rate:
                raise ValueError(
                    "AudioFrame sample_rate must match target_sample_rate "
                    f"{self._target_sample_rate}, got {msg.sample_rate}"
                )
            samples = self._audio_to_float(msg)
        except ValueError as exc:
            self.get_logger().error("Dropping invalid AudioFrame: %s", exc)
            return
        if samples.size == 0:
            return

        pcm_bytes = _float_to_pcm16(samples)
        try:
            probability, is_speech, start, end = self._vad.update(pcm_bytes)
        except Exception as exc:
            self.get_logger().fatal("VAD backend failed: %s", exc)
            rclpy.shutdown()
            raise

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
        return audio_frame_to_float_samples(
            data=bytes(msg.data),
            source_id=msg.source_id,
            stream_id=msg.stream_id,
            layout=msg.layout,
            channels=int(msg.channels),
            bit_depth=int(msg.bit_depth),
        )


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
