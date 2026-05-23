#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AudioFrame, VoiceActivity
from fa_vad_py.audio_frame import AudioFrameContract, audio_frame_to_float32_mono
from fa_vad_py.backends.base import VadBackendSettings
from fa_vad_py.backends.factory import build_vad_backend
from fa_vad_py.speech_activity import (
    SpeechActivityConfig,
    SpeechActivityEstimator,
    VoiceActivitySnapshot,
)
from fa_vad_py.vad_probability_stream import VadProbabilityStream


class FaVadNode(Node):
    def __init__(self, *, parameter_overrides: Iterable[Parameter] | None = None) -> None:
        if parameter_overrides is None:
            super().__init__("fa_vad")
        else:
            super().__init__("fa_vad", parameter_overrides=list(parameter_overrides))

        self._declare_parameters()
        self._load_parameters()
        self.probability_stream = self._build_probability_stream()
        self.activity_estimator = self._build_activity_estimator()

        self.activity_pub = self.create_publisher(
            VoiceActivity,
            self.output_topic,
            self._qos_profile("output.qos.depth", "output.qos.reliable"),
        )
        self.audio_sub = self.create_subscription(
            AudioFrame,
            self.audio_topic,
            self.on_audio,
            self._qos_profile("audio.qos.depth", "audio.qos.reliable"),
        )
        self.get_logger().info(
            "fa_vad started: "
            f"audio={self.audio_topic} "
            f"output={self.output_topic} "
            f"source={self.contract.source_id} "
            f"stream={self.contract.stream_id} "
            f"backend.name={self.backend_name}"
        )

    def _declare_parameters(self) -> None:
        for name, parameter_type in (
            ("audio_topic", Parameter.Type.STRING),
            ("output_topic", Parameter.Type.STRING),
            ("expected_source_id", Parameter.Type.STRING),
            ("expected_stream_id", Parameter.Type.STRING),
            ("expected.sample_rate", Parameter.Type.INTEGER),
            ("expected.channels", Parameter.Type.INTEGER),
            ("expected.encoding", Parameter.Type.STRING),
            ("expected.bit_depth", Parameter.Type.INTEGER),
            ("expected.layout", Parameter.Type.STRING),
            ("backend.name", Parameter.Type.STRING),
            ("backend.model_path", Parameter.Type.STRING),
            ("backend.execution_provider", Parameter.Type.STRING),
            ("backend.inter_op_num_threads", Parameter.Type.INTEGER),
            ("backend.intra_op_num_threads", Parameter.Type.INTEGER),
            ("detector.speech_threshold", Parameter.Type.DOUBLE),
            ("detector.silence_threshold", Parameter.Type.DOUBLE),
            ("detector.start_delta", Parameter.Type.DOUBLE),
            ("detector.end_delta", Parameter.Type.DOUBLE),
            ("detector.min_start_probability", Parameter.Type.DOUBLE),
            ("detector.max_end_probability", Parameter.Type.DOUBLE),
            ("detector.smoothing_alpha", Parameter.Type.DOUBLE),
            ("detector.start_consecutive_windows", Parameter.Type.INTEGER),
            ("detector.end_consecutive_windows", Parameter.Type.INTEGER),
            ("detector.min_speech_ms", Parameter.Type.INTEGER),
            ("audio.qos.depth", Parameter.Type.INTEGER),
            ("audio.qos.reliable", Parameter.Type.BOOL),
            ("output.qos.depth", Parameter.Type.INTEGER),
            ("output.qos.reliable", Parameter.Type.BOOL),
        ):
            self.declare_parameter(name, parameter_type)

    def _load_parameters(self) -> None:
        self.audio_topic = self._required_string("audio_topic")
        self.output_topic = self._required_string("output_topic")
        self.backend_name = self._required_string("backend.name")
        self.contract = AudioFrameContract(
            source_id=self._required_string("expected_source_id"),
            stream_id=self._required_string("expected_stream_id"),
            sample_rate=self._positive_int("expected.sample_rate"),
            channels=self._positive_int("expected.channels"),
            encoding=self._required_string("expected.encoding"),
            bit_depth=self._positive_int("expected.bit_depth"),
            layout=self._required_string("expected.layout"),
        )
        if self.contract.channels != 1:
            raise RuntimeError("fa_vad requires expected.channels=1")
        if self.contract.encoding != "FLOAT32LE":
            raise RuntimeError("fa_vad requires expected.encoding=FLOAT32LE")
        if self.contract.bit_depth != 32:
            raise RuntimeError("fa_vad requires expected.bit_depth=32")
        if self.contract.layout != "interleaved":
            raise RuntimeError("fa_vad requires expected.layout=interleaved")

    def _build_probability_stream(self) -> VadProbabilityStream:
        backend = build_vad_backend(
            VadBackendSettings(
                name=self.backend_name,
                model_path=self._required_string("backend.model_path"),
                sample_rate=self.contract.sample_rate,
                execution_provider=self._required_string("backend.execution_provider"),
                inter_op_num_threads=self._positive_int("backend.inter_op_num_threads"),
                intra_op_num_threads=self._positive_int("backend.intra_op_num_threads"),
            )
        )
        return VadProbabilityStream(backend)

    def _build_activity_estimator(self) -> SpeechActivityEstimator:
        return SpeechActivityEstimator(
            SpeechActivityConfig(
                speech_threshold=self._double("detector.speech_threshold"),
                silence_threshold=self._double("detector.silence_threshold"),
                start_delta=self._double("detector.start_delta"),
                end_delta=self._double("detector.end_delta"),
                min_start_probability=self._double("detector.min_start_probability"),
                max_end_probability=self._double("detector.max_end_probability"),
                smoothing_alpha=self._double("detector.smoothing_alpha"),
                start_consecutive_windows=self._positive_int(
                    "detector.start_consecutive_windows"
                ),
                end_consecutive_windows=self._positive_int("detector.end_consecutive_windows"),
                min_speech_ms=self._non_negative_int("detector.min_speech_ms"),
                sample_rate=self.contract.sample_rate,
            )
        )

    def on_audio(self, msg: AudioFrame) -> None:
        try:
            samples = audio_frame_to_float32_mono(msg, self.contract)
            probability_frames = self.probability_stream.push(samples)
            snapshots = [
                self.activity_estimator.update(frame)
                for frame in probability_frames
            ]
        except ValueError as exc:
            self.get_logger().warning(f"Dropping AudioFrame before VAD: {exc}")
            return
        except Exception as exc:
            self.get_logger().fatal(f"VAD backend failed: {exc}")
            rclpy.shutdown()
            raise

        for snapshot in snapshots:
            self.activity_pub.publish(self._to_msg(msg, snapshot))

    def _to_msg(self, audio_msg: AudioFrame, snapshot: VoiceActivitySnapshot) -> VoiceActivity:
        out = VoiceActivity()
        out.header = audio_msg.header
        out.source_id = audio_msg.source_id
        out.stream_id = audio_msg.stream_id
        out.sample_rate = int(audio_msg.sample_rate)
        out.epoch = int(audio_msg.epoch)
        out.window_start_sample = int(snapshot.window_start_sample)
        out.window_end_sample = int(snapshot.window_end_sample)
        out.probability = float(snapshot.probability)
        out.smoothed_probability = float(snapshot.smoothed_probability)
        out.is_speech = bool(snapshot.is_speech)
        out.speech_started = bool(snapshot.speech_started)
        out.speech_ended = bool(snapshot.speech_ended)
        return out

    def _parameter(self, name: str) -> Parameter:
        try:
            return self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc

    def _required_string(self, name: str) -> str:
        parameter = self._parameter(name)
        if parameter.type_ != Parameter.Type.STRING:
            raise RuntimeError(f"{name} must be a string")
        value = str(parameter.value).strip()
        if not value:
            raise RuntimeError(f"{name} is required")
        return value

    def _bool(self, name: str) -> bool:
        parameter = self._parameter(name)
        if parameter.type_ != Parameter.Type.BOOL:
            raise RuntimeError(f"{name} must be a bool")
        return bool(parameter.value)

    def _int(self, name: str) -> int:
        parameter = self._parameter(name)
        if parameter.type_ != Parameter.Type.INTEGER:
            raise RuntimeError(f"{name} must be an integer")
        return int(parameter.value)

    def _positive_int(self, name: str) -> int:
        value = self._int(name)
        if value <= 0:
            raise RuntimeError(f"{name} must be > 0")
        return value

    def _non_negative_int(self, name: str) -> int:
        value = self._int(name)
        if value < 0:
            raise RuntimeError(f"{name} must be >= 0")
        return value

    def _double(self, name: str) -> float:
        parameter = self._parameter(name)
        if parameter.type_ != Parameter.Type.DOUBLE:
            raise RuntimeError(f"{name} must be a double")
        return float(parameter.value)

    def _qos_profile(self, depth_parameter: str, reliable_parameter: str) -> QoSProfile:
        qos = QoSProfile(depth=self._positive_int(depth_parameter))
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (
            ReliabilityPolicy.RELIABLE
            if self._bool(reliable_parameter)
            else ReliabilityPolicy.BEST_EFFORT
        )
        return qos


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FaVadNode()
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
