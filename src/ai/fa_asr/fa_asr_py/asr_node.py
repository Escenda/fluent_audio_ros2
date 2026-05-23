#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header

from fa_asr_py.asr_stream_controller import (
    AsrStreamConfig,
    AsrStreamController,
    AsrTranscriptEvent,
)
from fa_asr_py.audio_frame import AudioFrameContract, audio_frame_to_float32_mono
from fa_asr_py.backends.parakeet_rnnt_stream_processor import (
    ParakeetRnntStreamProcessor,
    ParakeetStreamConfig,
)
from fa_interfaces.msg import AsrControl, AsrTranscript, AudioFrame


class FaAsrNode(Node):
    def __init__(self, *, parameter_overrides: Iterable[Parameter] | None = None) -> None:
        if parameter_overrides is None:
            super().__init__("fa_asr")
        else:
            super().__init__("fa_asr", parameter_overrides=list(parameter_overrides))

        self._declare_parameters()
        self._load_parameters()
        self.stream_controller = self._build_stream_controller()
        self._last_header = Header()
        self._last_epoch = 0

        self.transcript_pub = self.create_publisher(
            AsrTranscript,
            self.output_topic,
            self._qos_profile("output.qos.depth", "output.qos.reliable"),
        )
        self.audio_sub = self.create_subscription(
            AudioFrame,
            self.audio_topic,
            self.on_audio,
            self._qos_profile("audio.qos.depth", "audio.qos.reliable"),
        )
        self.control_sub = self.create_subscription(
            AsrControl,
            self.control_topic,
            self.on_control,
            self._qos_profile("control.qos.depth", "control.qos.reliable"),
        )

        self.get_logger().info(
            "fa_asr started: "
            f"audio={self.audio_topic} "
            f"control={self.control_topic} "
            f"output={self.output_topic} "
            f"backend.name={self.backend_name}"
        )

    def _declare_parameters(self) -> None:
        for name, parameter_type in (
            ("audio_topic", Parameter.Type.STRING),
            ("control_topic", Parameter.Type.STRING),
            ("output_topic", Parameter.Type.STRING),
            ("expected_source_id", Parameter.Type.STRING),
            ("expected_stream_id", Parameter.Type.STRING),
            ("expected.sample_rate", Parameter.Type.INTEGER),
            ("expected.channels", Parameter.Type.INTEGER),
            ("expected.encoding", Parameter.Type.STRING),
            ("expected.bit_depth", Parameter.Type.INTEGER),
            ("expected.layout", Parameter.Type.STRING),
            ("backend.name", Parameter.Type.STRING),
            ("backend.model_id", Parameter.Type.STRING),
            ("backend.model_path", Parameter.Type.STRING),
            ("backend.device", Parameter.Type.STRING),
            ("backend.compute_dtype", Parameter.Type.STRING),
            ("backend.left_context_secs", Parameter.Type.DOUBLE),
            ("backend.chunk_secs", Parameter.Type.DOUBLE),
            ("backend.right_context_secs", Parameter.Type.DOUBLE),
            ("backend.att_context_size_as_chunk", Parameter.Type.BOOL),
            ("controller.preroll_ms", Parameter.Type.INTEGER),
            ("audio.qos.depth", Parameter.Type.INTEGER),
            ("audio.qos.reliable", Parameter.Type.BOOL),
            ("control.qos.depth", Parameter.Type.INTEGER),
            ("control.qos.reliable", Parameter.Type.BOOL),
            ("output.qos.depth", Parameter.Type.INTEGER),
            ("output.qos.reliable", Parameter.Type.BOOL),
        ):
            self.declare_parameter(name, parameter_type)

    def _load_parameters(self) -> None:
        self.audio_topic = self._required_string("audio_topic")
        self.control_topic = self._required_string("control_topic")
        self.output_topic = self._required_string("output_topic")
        self.backend_name = self._required_string("backend.name")
        self.model_id = self._string("backend.model_id").strip() or self.backend_name
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
            raise RuntimeError("fa_asr requires expected.channels=1")
        if self.contract.encoding != "FLOAT32LE":
            raise RuntimeError("fa_asr requires expected.encoding=FLOAT32LE")
        if self.contract.bit_depth != 32:
            raise RuntimeError("fa_asr requires expected.bit_depth=32")
        if self.contract.layout != "interleaved":
            raise RuntimeError("fa_asr requires expected.layout=interleaved")

    def _build_stream_controller(self) -> AsrStreamController:
        processor = self._build_processor()
        return AsrStreamController(
            processor,
            AsrStreamConfig(
                sample_rate=self.contract.sample_rate,
                preroll_ms=self._non_negative_int("controller.preroll_ms"),
            ),
        )

    def _build_processor(self) -> ParakeetRnntStreamProcessor:
        if self.backend_name != "parakeet_rnnt_streaming":
            raise RuntimeError(f"unsupported ASR backend.name: {self.backend_name}")
        return ParakeetRnntStreamProcessor(
            ParakeetStreamConfig(
                model_path=self._required_string("backend.model_path"),
                device=self._required_string("backend.device"),
                compute_dtype=self._required_string("backend.compute_dtype"),
                sample_rate=self.contract.sample_rate,
                left_context_secs=self._double("backend.left_context_secs"),
                chunk_secs=self._double("backend.chunk_secs"),
                right_context_secs=self._double("backend.right_context_secs"),
                att_context_size_as_chunk=self._bool("backend.att_context_size_as_chunk"),
            )
        )

    def on_audio(self, msg: AudioFrame) -> None:
        try:
            samples = audio_frame_to_float32_mono(msg, self.contract)
            self._last_header = msg.header
            self._last_epoch = int(msg.epoch)
            events = self.stream_controller.on_audio(samples)
        except ValueError as exc:
            self.get_logger().warning(f"Dropping AudioFrame before ASR: {exc}")
            return
        except Exception as exc:
            self.get_logger().fatal(f"ASR backend failed: {exc}")
            rclpy.shutdown()
            raise
        self._publish_events(
            events,
            header=msg.header,
            source_id=msg.source_id,
            stream_id=msg.stream_id,
            epoch=int(msg.epoch),
        )

    def on_control(self, msg: AsrControl) -> None:
        before_state = self.stream_controller.state
        try:
            if int(msg.action) == AsrControl.ACTION_START:
                events = self.stream_controller.start(
                    session_id=str(msg.session_id),
                    user_turn_id=int(msg.user_turn_id),
                )
            elif int(msg.action) == AsrControl.ACTION_STOP:
                events = self.stream_controller.stop()
            elif int(msg.action) == AsrControl.ACTION_CANCEL:
                events = self.stream_controller.cancel()
            else:
                self.get_logger().warning(f"Dropping unknown ASR control action: {msg.action}")
                return
        except ValueError as exc:
            self.get_logger().warning(f"Dropping invalid ASR control: {exc}")
            return
        self._log_control(msg, before_state)
        self._publish_events(
            events,
            header=self._header_from_time(msg.timestamp),
            source_id=self.contract.source_id,
            stream_id=self.contract.stream_id,
            epoch=self._last_epoch,
        )

    def _log_control(self, msg: AsrControl, before_state: object) -> None:
        after_state = self.stream_controller.state
        before = getattr(before_state, "value", str(before_state))
        after = getattr(after_state, "value", str(after_state))
        action = self._action_name(int(msg.action))
        self.get_logger().info(
            "ASR control: "
            f"action={action} state={before}->{after} "
            f"session={msg.session_id or '-'} turn={int(msg.user_turn_id)} "
            f"reason={msg.reason or '-'}"
        )

    @staticmethod
    def _action_name(action: int) -> str:
        if action == AsrControl.ACTION_START:
            return "START"
        if action == AsrControl.ACTION_STOP:
            return "STOP"
        if action == AsrControl.ACTION_CANCEL:
            return "CANCEL"
        return f"UNKNOWN({action})"

    def _publish_events(
        self,
        events: list[AsrTranscriptEvent],
        *,
        header: Header,
        source_id: str,
        stream_id: str,
        epoch: int,
    ) -> None:
        for event in events:
            out = AsrTranscript()
            out.header = header
            out.source_id = source_id
            out.stream_id = stream_id
            out.epoch = int(epoch)
            out.session_id = event.session_id
            out.user_turn_id = int(event.user_turn_id)
            out.text = event.text
            out.is_final = bool(event.is_final)
            out.accepted_samples = int(event.accepted_samples)
            out.model_id = self.model_id
            self.transcript_pub.publish(out)

    @staticmethod
    def _header_from_time(timestamp: object) -> Header:
        header = Header()
        header.stamp = timestamp
        return header

    def _parameter(self, name: str) -> Parameter:
        try:
            return self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc

    def _string(self, name: str) -> str:
        parameter = self._parameter(name)
        if parameter.type_ != Parameter.Type.STRING:
            raise RuntimeError(f"{name} must be a string")
        return str(parameter.value)

    def _required_string(self, name: str) -> str:
        value = self._string(name).strip()
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
    node = FaAsrNode()
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
