#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TextIO, cast

import numpy as np
from builtin_interfaces.msg import Time
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import (
    AsrEvent,
    AsrResult,
    AsrState,
    AudioFrame,
    AudioModelRef,
    AudioWindowRef,
    ResolvedTimeRange,
    TranscriptSegment,
    TurnContext,
    VadState,
)
from fa_interfaces.srv import TranscribeAudio
from fa_asr_py.backends.base import (
    AsrBackend,
    AsrBackendCapability,
    AsrAudioPayload,
    AsrRequest,
    AsrStreamRequest,
    AsrStreamResult,
    AsrStreamingSession,
    StreamingAsrBackend,
    AsrTranscript,
    asr_transcript_text,
    build_asr_transcript,
)
from fa_asr_py.backends.factory import AsrBackendSettings, build_asr_backend
from fa_asr_py.timeline import (
    ERROR_RANGE_NOT_CONTINUOUS,
    ERROR_TIME_RANGE_UNRESOLVED,
    NumericTimeRange,
    RollingAsrTimeline,
    TimelineRangeError,
    TimelineSlice,
    parse_numeric_time_range,
)


_NSEC_PER_SEC = 1_000_000_000


@dataclass(frozen=True)
class TranscriptionJob:
    session_id: str
    user_turn_id: int
    payload: AsrAudioPayload
    reason: str


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
    open_on: str
    close_on: str
    submit_on_close: bool
    pre_roll_ms: float
    post_roll_ms: float
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
    stamp_unix_ns: int


@dataclass
class ControlWindowState:
    open: bool = False
    start_unix_ns: int = 0
    previous_active: bool = False
    stream_session: AsrStreamingSession | None = None
    stream_session_id: str = ""
    stream_user_turn_id: int = 0
    stream_sample_count: int = 0


class FaAsrNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_asr")

        self._declare_required_parameters()

        self.audio_topic = self._string_parameter("audio_topic").strip()
        self.turn_context_topic = self._string_parameter("turn_context_topic").strip()
        self.asr_result_topic = self._string_parameter("asr_result_topic").strip()
        self.asr_state_topic = self._string_parameter("asr_state_topic").strip()
        self.asr_event_topic = self._string_parameter("asr_event_topic").strip()
        self.transcribe_service_name = self._string_parameter(
            "transcribe_service_name"
        ).strip()
        self.expected_source_id = self._string_parameter("expected_source_id").strip()
        if not self.expected_source_id:
            raise RuntimeError("expected_source_id is required")
        self.expected_stream_id = self._string_parameter("expected_stream_id").strip()
        if not self.expected_stream_id:
            raise RuntimeError("expected_stream_id is required")
        self.target_sample_rate = self._positive_integer_parameter("target_sample_rate")
        self.min_audio_sec = self._positive_double_parameter("min_audio_sec")
        self.timeline_retention_sec = self._positive_double_parameter(
            "timeline.retention_sec"
        )
        self.timeline_timestamp_alignment_tolerance_ms = (
            self._non_negative_double_parameter(
                "timeline.timestamp_alignment_tolerance_ms"
            )
        )
        self.timeline_clock = self._timeline_clock_parameter("timeline.clock")
        self.timeline_window_id = self._string_parameter("timeline.window_id").strip()
        if not self.timeline_window_id:
            raise RuntimeError("timeline.window_id is required")
        self.timeline_window_epoch = self._non_negative_integer_parameter(
            "timeline.window_epoch"
        )
        self.silence_timeout_sec = self._positive_double_parameter("silence_timeout_sec")
        self.control_default_enabled = self._bool_parameter("control.default_enabled")
        self.control_configs = self._load_control_configs()
        self.finalize_on_context_inactive = self._bool_parameter(
            "finalize_on_context_inactive"
        )
        self.workspace_dir = Path(self._string_parameter("workspace_dir")).expanduser()
        self.cleanup_audio_files = self._bool_parameter("cleanup_audio_files")
        self.trace_enabled = self._bool_parameter("trace.enabled")
        self.trace_path = self._string_parameter("trace.path").strip()
        self._trace_lock = threading.Lock()
        self._trace_file = self._open_trace_file(
            enabled=self.trace_enabled,
            trace_path=self.trace_path,
        )
        self.backend_name = self._string_parameter("backend.name").strip()
        self.backend_kind = self._backend_kind_parameter("backend.kind")
        self.backend_model = self._string_parameter("backend.model").strip()
        self.backend_model_path = self._string_parameter("backend.model_path").strip()
        self.backend_model_version = self._string_parameter(
            "backend.model_version"
        ).strip()
        self.backend_model_revision = self._string_parameter(
            "backend.model_revision"
        ).strip()
        self._validate_identity_contract()

        self.backend = self._load_backend()
        self.input_capability = self._effective_input_capability(self.backend.capability)
        if self.input_capability.sample_rate_hz != self.target_sample_rate:
            raise RuntimeError(
                "backend capability sample_rate_hz must match target_sample_rate "
                f"{self.target_sample_rate}, got {self.input_capability.sample_rate_hz}"
            )
        self._backend_lock = threading.Lock()
        self._timeline = self._build_timeline()
        self._timeline_lock = threading.Lock()
        self._turn_state_lock = threading.RLock()
        self._io_callback_group = MutuallyExclusiveCallbackGroup()
        self._transcribe_service_callback_group = MutuallyExclusiveCallbackGroup()

        qos_audio = self._qos_profile(
            depth_parameter="audio.qos.depth",
            reliable_parameter="audio.qos.reliable",
        )
        qos_turn_context = self._qos_profile(
            depth_parameter="turn_context.qos.depth",
            reliable_parameter="turn_context.qos.reliable",
        )
        qos_result = self._qos_profile(
            depth_parameter="result.qos.depth",
            reliable_parameter="result.qos.reliable",
        )
        qos_observability = self._qos_profile(
            depth_parameter="observability.qos.depth",
            reliable_parameter="observability.qos.reliable",
        )

        self.asr_result_pub = self.create_publisher(
            AsrResult, self.asr_result_topic, qos_result
        )
        self.asr_state_pub = self.create_publisher(
            AsrState, self.asr_state_topic, qos_observability
        )
        self.asr_event_pub = self.create_publisher(
            AsrEvent, self.asr_event_topic, qos_observability
        )
        self.audio_sub = self.create_subscription(
            AudioFrame,
            self.audio_topic,
            self.on_audio,
            qos_audio,
            callback_group=self._io_callback_group,
        )
        self.control_subs = []
        self._control_windows = {
            config.control_id: ControlWindowState() for config in self.control_configs
        }
        for control_config in self.control_configs:
            self.control_subs.append(self._create_control_subscription(control_config))
        self.turn_context_sub = self.create_subscription(
            TurnContext,
            self.turn_context_topic,
            self.on_turn_context,
            qos_turn_context,
            callback_group=self._io_callback_group,
        )
        self.transcribe_srv = self.create_service(
            TranscribeAudio,
            self.transcribe_service_name,
            self.handle_transcribe_audio,
            callback_group=self._transcribe_service_callback_group,
        )
        self.timer = self.create_timer(
            0.5,
            self._check_timeout,
            callback_group=self._io_callback_group,
        )

        self._active_session_id = ""
        self._active_user_turn_id = 0
        self._context_active = False
        self._last_speech = self.get_clock().now()
        self._payload_chunks: list[bytes] = []
        self._buffer_sample_count = 0
        self._samples_lock = threading.Lock()
        self._jobs: queue.Queue[TranscriptionJob | None] = queue.Queue()
        self._fail_closed_triggered = False
        self._asr_state = AsrState.STATE_WAITING
        self._event_seq = 0
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._emit_event(
            AsrEvent.EVENT_STARTUP_IDLE,
            "startup_idle",
        )
        self.get_logger().info(
            f"fa_asr started: audio={self.audio_topic} "
            f"expected_source_id={self.expected_source_id} "
            f"expected_stream_id={self.expected_stream_id} "
            f"control_inputs={','.join(config.control_id for config in self.control_configs)} "
            f"turn_context={self.turn_context_topic} result={self.asr_result_topic} "
            f"transcribe_service={self.transcribe_service_name} "
            f"target_sr={self.target_sample_rate} backend.name={self.backend.name}"
        )

    def _declare_required_parameters(self) -> None:
        self.declare_parameter("audio_topic", Parameter.Type.STRING)
        self.declare_parameter("turn_context_topic", Parameter.Type.STRING)
        self.declare_parameter("asr_result_topic", Parameter.Type.STRING)
        self.declare_parameter("asr_state_topic", Parameter.Type.STRING)
        self.declare_parameter("asr_event_topic", Parameter.Type.STRING)
        self.declare_parameter("transcribe_service_name", Parameter.Type.STRING)
        self.declare_parameter("expected_source_id", Parameter.Type.STRING)
        self.declare_parameter("expected_stream_id", Parameter.Type.STRING)
        self.declare_parameter("target_sample_rate", Parameter.Type.INTEGER)
        self.declare_parameter("min_audio_sec", Parameter.Type.DOUBLE)
        self.declare_parameter("timeline.retention_sec", Parameter.Type.DOUBLE)
        self.declare_parameter(
            "timeline.timestamp_alignment_tolerance_ms",
            Parameter.Type.DOUBLE,
        )
        self.declare_parameter("timeline.clock", Parameter.Type.STRING)
        self.declare_parameter("timeline.window_id", Parameter.Type.STRING)
        self.declare_parameter("timeline.window_epoch", Parameter.Type.INTEGER)
        self.declare_parameter("silence_timeout_sec", Parameter.Type.DOUBLE)
        self.declare_parameter("control.default_enabled", Parameter.Type.BOOL)
        self.declare_parameter("control.inputs", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("finalize_on_context_inactive", Parameter.Type.BOOL)
        self.declare_parameter("workspace_dir", Parameter.Type.STRING)
        self.declare_parameter("cleanup_audio_files", Parameter.Type.BOOL)
        self.declare_parameter("trace.enabled", Parameter.Type.BOOL)
        self.declare_parameter("trace.path", Parameter.Type.STRING)
        self.declare_parameter("backend.name", Parameter.Type.STRING)
        self.declare_parameter("backend.kind", Parameter.Type.STRING)
        self.declare_parameter("backend.model", Parameter.Type.STRING)
        self.declare_parameter("backend.command", Parameter.Type.STRING)
        self.declare_parameter("backend.model_path", Parameter.Type.STRING)
        self.declare_parameter("backend.model_version", Parameter.Type.STRING)
        self.declare_parameter("backend.model_revision", Parameter.Type.STRING)
        self.declare_parameter("backend.openai_realtime.api_key_env", Parameter.Type.STRING)
        self.declare_parameter(
            "backend.openai_transcriptions.api_key_env",
            Parameter.Type.STRING,
        )
        self.declare_parameter("backend.language", Parameter.Type.STRING)
        self.declare_parameter("backend.timeout_sec", Parameter.Type.DOUBLE)
        self.declare_parameter("backend.working_directory", Parameter.Type.STRING)
        self.declare_parameter("backend.args", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("backend.health_args", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("backend.output_text_path", Parameter.Type.STRING)
        self.declare_parameter("backend.result_format", Parameter.Type.STRING)
        self.declare_parameter("audio.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("audio.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("turn_context.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("turn_context.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("result.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("result.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("observability.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("observability.qos.reliable", Parameter.Type.BOOL)

    def _validate_identity_contract(self) -> None:
        topics = (
            ("audio_topic", self.audio_topic),
            ("turn_context_topic", self.turn_context_topic),
            ("asr_result_topic", self.asr_result_topic),
            ("asr_state_topic", self.asr_state_topic),
            ("asr_event_topic", self.asr_event_topic),
            ("transcribe_service_name", self.transcribe_service_name),
        )
        for topic_name, topic_value in topics:
            if not topic_value:
                raise RuntimeError(f"{topic_name} is required")
            if self._same_identity_string(self.expected_stream_id, topic_value):
                raise RuntimeError(
                    f"expected_stream_id must be distinct from ROS {topic_name}"
                )
        if self._same_identity_string(self.audio_topic, self.asr_result_topic):
            raise RuntimeError("audio_topic must be distinct from asr_result_topic")

    def _load_control_configs(self) -> tuple[ControlInputConfig, ...]:
        control_ids = self._string_array_parameter("control.inputs")
        if len(control_ids) != len(set(control_ids)):
            raise RuntimeError("control.inputs must not contain duplicate IDs")

        configs: list[ControlInputConfig] = []
        for control_id in control_ids:
            normalized_id = control_id.strip()
            if not normalized_id:
                raise RuntimeError("control.inputs must not contain empty IDs")
            if normalized_id != control_id:
                raise RuntimeError("control.inputs IDs must not contain surrounding whitespace")
            prefix = f"control.{normalized_id}"
            self._declare_control_parameters(prefix)
            configs.append(
                ControlInputConfig(
                    control_id=normalized_id,
                    action=self._control_action_parameter(f"{prefix}.action"),
                    topic=self._required_string_parameter(f"{prefix}.topic"),
                    msg_type=self._control_msg_type_parameter(f"{prefix}.msg_type"),
                    source_id=self._required_string_parameter(f"{prefix}.source_id"),
                    stream_id=self._required_string_parameter(f"{prefix}.stream_id"),
                    active_field=self._vad_control_field_parameter(
                        f"{prefix}.active_field",
                        expected_field="is_speech",
                    ),
                    start_field=self._vad_control_field_parameter(
                        f"{prefix}.start_field",
                        expected_field="start",
                    ),
                    end_field=self._vad_control_field_parameter(
                        f"{prefix}.end_field",
                        expected_field="end",
                    ),
                    open_on=self._control_open_policy_parameter(f"{prefix}.open_on"),
                    close_on=self._control_close_policy_parameter(f"{prefix}.close_on"),
                    submit_on_close=self._bool_parameter(f"{prefix}.submit_on_close"),
                    pre_roll_ms=self._non_negative_double_parameter(f"{prefix}.pre_roll_ms"),
                    post_roll_ms=self._non_negative_double_parameter(f"{prefix}.post_roll_ms"),
                    qos_depth=self._positive_integer_parameter(f"{prefix}.qos.depth"),
                    qos_reliable=self._bool_parameter(f"{prefix}.qos.reliable"),
                )
            )
        return tuple(configs)

    def _build_timeline(self) -> RollingAsrTimeline:
        return RollingAsrTimeline(
            sample_rate=self.target_sample_rate,
            retention_sec=self.timeline_retention_sec,
            timestamp_alignment_tolerance_ms=self.timeline_timestamp_alignment_tolerance_ms,
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
        self.declare_parameter(f"{prefix}.open_on", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.close_on", Parameter.Type.STRING)
        self.declare_parameter(f"{prefix}.submit_on_close", Parameter.Type.BOOL)
        self.declare_parameter(f"{prefix}.pre_roll_ms", Parameter.Type.DOUBLE)
        self.declare_parameter(f"{prefix}.post_roll_ms", Parameter.Type.DOUBLE)
        self.declare_parameter(f"{prefix}.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter(f"{prefix}.qos.reliable", Parameter.Type.BOOL)

    @staticmethod
    def _same_identity_string(left: str, right: str) -> bool:
        return left.lstrip("/") == right.lstrip("/")

    def _load_backend(self) -> AsrBackend:
        backend_name = self._string_parameter("backend.name").strip()
        if backend_name in ("local_command", "whisper.cpp"):
            return build_asr_backend(
                AsrBackendSettings(
                    name=backend_name,
                    workspace_dir=self.workspace_dir,
                    cleanup_audio_files=self.cleanup_audio_files,
                    command=self._string_parameter("backend.command"),
                    model_path=self._string_parameter("backend.model_path"),
                    language=self._string_parameter("backend.language"),
                    args=self._backend_args(),
                    health_args=self._string_array_parameter("backend.health_args"),
                    timeout_sec=self._double_parameter("backend.timeout_sec"),
                    working_directory=self._string_parameter("backend.working_directory"),
                    output_text_path=self._string_parameter("backend.output_text_path"),
                    result_format=self._string_parameter("backend.result_format"),
                )
            )
        if backend_name == "parakeet_worker":
            return build_asr_backend(
                AsrBackendSettings(
                    name=backend_name,
                    workspace_dir=self.workspace_dir,
                    cleanup_audio_files=self.cleanup_audio_files,
                    command=self._string_parameter("backend.command"),
                    model=self._string_parameter("backend.model"),
                    language=self._string_parameter("backend.language"),
                    args=self._backend_args(),
                    health_args=self._string_array_parameter("backend.health_args"),
                    timeout_sec=self._double_parameter("backend.timeout_sec"),
                    working_directory=self._string_parameter("backend.working_directory"),
                    output_text_path=self._string_parameter("backend.output_text_path"),
                    result_format=self._string_parameter("backend.result_format"),
                )
            )
        if backend_name == "openai_realtime":
            return build_asr_backend(
                AsrBackendSettings(
                    name=backend_name,
                    workspace_dir=self.workspace_dir,
                    cleanup_audio_files=self.cleanup_audio_files,
                    command=self._string_parameter("backend.command"),
                    model=self._string_parameter("backend.model"),
                    openai_realtime_api_key_env=self._string_parameter(
                        "backend.openai_realtime.api_key_env"
                    ),
                    language=self._string_parameter("backend.language"),
                    args=self._backend_args(),
                    health_args=self._string_array_parameter("backend.health_args"),
                    timeout_sec=self._double_parameter("backend.timeout_sec"),
                    working_directory=self._string_parameter("backend.working_directory"),
                    output_text_path=self._string_parameter("backend.output_text_path"),
                    result_format=self._string_parameter("backend.result_format"),
                )
            )
        if backend_name == "openai_transcriptions":
            return build_asr_backend(
                AsrBackendSettings(
                    name=backend_name,
                    workspace_dir=self.workspace_dir,
                    cleanup_audio_files=self.cleanup_audio_files,
                    command=self._string_parameter("backend.command"),
                    model=self._string_parameter("backend.model"),
                    openai_transcriptions_api_key_env=self._string_parameter(
                        "backend.openai_transcriptions.api_key_env"
                    ),
                    language=self._string_parameter("backend.language"),
                    args=self._backend_args(),
                    health_args=self._string_array_parameter("backend.health_args"),
                    timeout_sec=self._double_parameter("backend.timeout_sec"),
                    working_directory=self._string_parameter("backend.working_directory"),
                    output_text_path=self._string_parameter("backend.output_text_path"),
                    result_format=self._string_parameter("backend.result_format"),
                )
            )
        return build_asr_backend(
            AsrBackendSettings(
                name=backend_name,
                workspace_dir=self.workspace_dir,
                cleanup_audio_files=self.cleanup_audio_files,
            )
        )

    def _effective_input_capability(
        self,
        capability: AsrBackendCapability,
    ) -> AsrBackendCapability:
        if capability.sample_rate_hz > 0:
            return capability
        return AsrBackendCapability(
            audio_encoding=capability.audio_encoding,
            sample_rate_hz=self.target_sample_rate,
            channels=capability.channels,
            streaming=capability.streaming,
            final_results_only=capability.final_results_only,
        )

    def _streaming_backend(self) -> StreamingAsrBackend:
        if not self.input_capability.streaming:
            raise RuntimeError("ASR backend is not configured as streaming")
        if not hasattr(self.backend, "start_stream"):
            raise RuntimeError(
                "streaming ASR backend must implement start_stream(request)"
            )
        return cast(StreamingAsrBackend, self.backend)

    def _backend_args(self) -> tuple[str, ...]:
        return self._string_array_parameter("backend.args")

    def _string_parameter(self, name: str) -> str:
        try:
            parameter = self.get_parameter(name)
            value = parameter.value
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.STRING:
            raise RuntimeError(f"{name} must be a string")
        return value

    def _bool_parameter(self, name: str) -> bool:
        try:
            parameter = self.get_parameter(name)
            value = parameter.value
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.BOOL:
            raise RuntimeError(f"{name} must be a bool")
        return value

    def _required_string_parameter(self, name: str) -> str:
        value = FaAsrNode._string_parameter(self, name).strip()
        if not value:
            raise RuntimeError(f"{name} is required")
        return value

    def _integer_parameter(self, name: str) -> int:
        try:
            parameter = self.get_parameter(name)
            value = parameter.value
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.INTEGER:
            raise RuntimeError(f"{name} must be an integer")
        return value

    def _positive_integer_parameter(self, name: str) -> int:
        value = FaAsrNode._integer_parameter(self, name)
        if value <= 0:
            raise RuntimeError(f"{name} must be greater than zero")
        return value

    def _non_negative_integer_parameter(self, name: str) -> int:
        value = FaAsrNode._integer_parameter(self, name)
        if value < 0:
            raise RuntimeError(f"{name} must be greater than or equal to zero")
        return value

    def _double_parameter(self, name: str) -> float:
        try:
            parameter = self.get_parameter(name)
            value = parameter.value
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.DOUBLE:
            raise RuntimeError(f"{name} must be a double")
        return value

    def _positive_double_parameter(self, name: str) -> float:
        value = FaAsrNode._double_parameter(self, name)
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"{name} must be finite and greater than zero")
        return value

    def _non_negative_double_parameter(self, name: str) -> float:
        value = FaAsrNode._double_parameter(self, name)
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"{name} must be finite and greater than or equal to zero")
        return value

    def _timeline_clock_parameter(self, name: str) -> str:
        value = FaAsrNode._string_parameter(self, name).strip()
        if value not in (
            ResolvedTimeRange.CLOCK_AGENT,
            ResolvedTimeRange.CLOCK_ROBOT,
            ResolvedTimeRange.CLOCK_MEDIA,
        ):
            raise RuntimeError(f"{name} must be agent, robot, or media")
        return value

    def _backend_kind_parameter(self, name: str) -> str:
        value = FaAsrNode._string_parameter(self, name)
        if value != "asr":
            raise RuntimeError(f"{name} must be asr")
        return value

    def _control_action_parameter(self, name: str) -> str:
        value = FaAsrNode._string_parameter(self, name).strip()
        if value != "topic":
            raise RuntimeError(f"{name} must be topic")
        return value

    def _control_msg_type_parameter(self, name: str) -> str:
        value = FaAsrNode._string_parameter(self, name).strip()
        if value != "fa_interfaces/msg/VadState":
            raise RuntimeError(f"{name} must be fa_interfaces/msg/VadState")
        return value

    def _vad_control_field_parameter(self, name: str, *, expected_field: str) -> str:
        value = FaAsrNode._string_parameter(self, name).strip()
        if value != expected_field:
            raise RuntimeError(f"{name} must be {expected_field}")
        return value

    def _control_open_policy_parameter(self, name: str) -> str:
        value = FaAsrNode._string_parameter(self, name).strip()
        if value != "start_or_active_rising":
            raise RuntimeError(f"{name} must be start_or_active_rising")
        return value

    def _control_close_policy_parameter(self, name: str) -> str:
        value = FaAsrNode._string_parameter(self, name).strip()
        if value != "end_or_active_falling":
            raise RuntimeError(f"{name} must be end_or_active_falling")
        return value

    def _qos_profile(self, *, depth_parameter: str, reliable_parameter: str) -> QoSProfile:
        depth = FaAsrNode._positive_integer_parameter(self, depth_parameter)
        reliable = FaAsrNode._bool_parameter(self, reliable_parameter)
        qos = QoSProfile(depth=depth)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (
            ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
        )
        return qos

    @staticmethod
    def _open_trace_file(*, enabled: bool, trace_path: str) -> TextIO | None:
        if not enabled:
            return None
        if not trace_path:
            raise RuntimeError("trace.path is required when trace.enabled is true")
        path = Path(trace_path).expanduser()
        try:
            return path.open("a", encoding="utf-8", buffering=1)
        except OSError as exc:
            raise RuntimeError(f"trace.path is not writable: {path}") from exc

    def _emit_event(
        self,
        event: int,
        reason: str,
        *,
        transition_to_state: int | None = None,
        session_id: str = "",
        user_turn_id: int = 0,
        control_id: str = "",
        source_id: str = "",
        stream_id: str = "",
        context_active: bool | None = None,
        window_open: bool = False,
        sample_count: int = 0,
        start_unix_ns: int = 0,
        end_unix_ns: int = 0,
        error_code: str = "",
    ) -> None:
        state_before = int(getattr(self, "_asr_state", AsrState.STATE_WAITING))
        state_after = int(
            transition_to_state if transition_to_state is not None else state_before
        )
        if transition_to_state is not None:
            self._asr_state = state_after
        event_seq = int(getattr(self, "_event_seq", 0)) + 1
        self._event_seq = event_seq
        active = (
            bool(context_active)
            if context_active is not None
            else bool(getattr(self, "_context_active", False))
        )
        queued_jobs = self._queued_job_count()
        timestamp = self.get_clock().now().to_msg()

        event_msg = AsrEvent()
        event_msg.timestamp = timestamp
        event_msg.event_seq = event_seq
        event_msg.event = int(event)
        event_msg.state = state_after
        event_msg.state_before = state_before
        event_msg.state_after = state_after
        event_msg.reason = reason
        event_msg.session_id = session_id
        event_msg.user_turn_id = int(user_turn_id)
        event_msg.control_id = control_id
        event_msg.source_id = source_id
        event_msg.stream_id = stream_id
        event_msg.context_active = active
        event_msg.window_open = window_open
        event_msg.queued_jobs = queued_jobs
        event_msg.sample_count = int(sample_count)
        event_msg.start_unix_ns = int(start_unix_ns)
        event_msg.end_unix_ns = int(end_unix_ns)
        event_msg.error_code = error_code
        event_pub = getattr(self, "asr_event_pub", None)
        if event_pub is not None:
            event_pub.publish(event_msg)

        self._publish_state(
            timestamp=timestamp,
            state=state_after,
            reason=reason,
            session_id=session_id,
            user_turn_id=user_turn_id,
            control_id=control_id,
            source_id=source_id,
            stream_id=stream_id,
            context_active=active,
            window_open=window_open,
            queued_jobs=queued_jobs,
            sample_count=sample_count,
            start_unix_ns=start_unix_ns,
            end_unix_ns=end_unix_ns,
            error_code=error_code,
        )
        self._write_trace_record(
            timestamp_sec=int(timestamp.sec),
            timestamp_nanosec=int(timestamp.nanosec),
            event_seq=event_seq,
            event=event,
            state=state_after,
            state_before=state_before,
            state_after=state_after,
            reason=reason,
            session_id=session_id,
            user_turn_id=user_turn_id,
            control_id=control_id,
            source_id=source_id,
            stream_id=stream_id,
            context_active=active,
            window_open=window_open,
            queued_jobs=queued_jobs,
            sample_count=sample_count,
            start_unix_ns=start_unix_ns,
            end_unix_ns=end_unix_ns,
            error_code=error_code,
        )

    def _publish_state(
        self,
        *,
        timestamp: Time,
        state: int,
        reason: str,
        session_id: str,
        user_turn_id: int,
        control_id: str,
        source_id: str,
        stream_id: str,
        context_active: bool,
        window_open: bool,
        queued_jobs: int,
        sample_count: int,
        start_unix_ns: int,
        end_unix_ns: int,
        error_code: str,
    ) -> None:
        state_msg = AsrState()
        state_msg.timestamp = timestamp
        state_msg.state = int(state)
        state_msg.reason = reason
        state_msg.session_id = session_id
        state_msg.user_turn_id = int(user_turn_id)
        state_msg.control_id = control_id
        state_msg.source_id = source_id
        state_msg.stream_id = stream_id
        state_msg.context_active = context_active
        state_msg.window_open = window_open
        state_msg.queued_jobs = int(queued_jobs)
        state_msg.sample_count = int(sample_count)
        state_msg.start_unix_ns = int(start_unix_ns)
        state_msg.end_unix_ns = int(end_unix_ns)
        state_msg.error_code = error_code
        state_pub = getattr(self, "asr_state_pub", None)
        if state_pub is not None:
            state_pub.publish(state_msg)

    def _write_trace_record(
        self,
        *,
        timestamp_sec: int,
        timestamp_nanosec: int,
        event_seq: int,
        event: int,
        state: int,
        state_before: int,
        state_after: int,
        reason: str,
        session_id: str,
        user_turn_id: int,
        control_id: str,
        source_id: str,
        stream_id: str,
        context_active: bool,
        window_open: bool,
        queued_jobs: int,
        sample_count: int,
        start_unix_ns: int,
        end_unix_ns: int,
        error_code: str,
    ) -> None:
        trace_file = getattr(self, "_trace_file", None)
        if trace_file is None:
            return
        line = json.dumps(
            {
                "timestamp_sec": int(timestamp_sec),
                "timestamp_nanosec": int(timestamp_nanosec),
                "event_seq": int(event_seq),
                "event": int(event),
                "state": int(state),
                "state_before": int(state_before),
                "state_after": int(state_after),
                "reason": reason,
                "session_id": session_id,
                "user_turn_id": int(user_turn_id),
                "control_id": control_id,
                "source_id": source_id,
                "stream_id": stream_id,
                "context_active": context_active,
                "window_open": window_open,
                "queued_jobs": int(queued_jobs),
                "sample_count": int(sample_count),
                "start_unix_ns": int(start_unix_ns),
                "end_unix_ns": int(end_unix_ns),
                "error_code": error_code,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._trace_lock:
            trace_file.write(f"{line}\n")

    def _queued_job_count(self) -> int:
        jobs = getattr(self, "_jobs", None)
        if jobs is None:
            return 0
        return int(jobs.qsize())

    def _string_array_parameter(self, name: str) -> tuple[str, ...]:
        try:
            parameter = self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        if parameter.type_ != Parameter.Type.STRING_ARRAY:
            raise RuntimeError(f"{name} must be a string array")
        try:
            array_value = parameter.get_parameter_value().string_array_value
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc
        return tuple(array_value)

    def _create_control_subscription(self, config: ControlInputConfig):
        if config.action != "topic":
            raise RuntimeError(f"control.{config.control_id}.action must be topic")
        if config.msg_type != "fa_interfaces/msg/VadState":
            raise RuntimeError(
                f"control.{config.control_id}.msg_type must be fa_interfaces/msg/VadState"
            )
        qos = self._qos_from_values(depth=config.qos_depth, reliable=config.qos_reliable)
        return self.create_subscription(
            VadState,
            config.topic,
            self._make_vad_control_callback(config),
            qos,
            callback_group=self._io_callback_group,
        )

    def _make_vad_control_callback(
        self,
        config: ControlInputConfig,
    ) -> Callable[[VadState], None]:
        def callback(msg: VadState) -> None:
            try:
                event = self._vad_state_to_control_event(config, msg)
            except TimelineRangeError as exc:
                self.get_logger().error(
                    f"Dropping control event {config.control_id}: {exc}"
                )
                return
            self.on_control_event(event)

        return callback

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
        with self._turn_state_lock:
            if not msg.active or not msg.session_id:
                if self.input_capability.streaming:
                    self._cancel_open_streams("context_inactive")
                if (
                    self.control_default_enabled
                    and self._context_active
                    and not self.input_capability.streaming
                    and self.finalize_on_context_inactive
                ):
                    self._submit_current_buffer(
                        "context_inactive",
                        publish_timeout_if_empty=False,
                    )
                self._context_active = False
                self._active_session_id = ""
                self._active_user_turn_id = 0
                self._clear_buffer()
                return

            new_session_id = str(msg.session_id)
            new_user_turn_id = int(msg.user_turn_id)
            key_changed = (
                self._active_session_id != new_session_id
                or self._active_user_turn_id != new_user_turn_id
            )
            if self.input_capability.streaming and self._context_active and key_changed:
                self._cancel_open_streams("turn_replaced")
            if (
                self.control_default_enabled
                and self._context_active
                and key_changed
                and not self.input_capability.streaming
            ):
                self._submit_current_buffer("turn_replaced", publish_timeout_if_empty=False)

            if key_changed:
                self._clear_buffer()
                self._last_speech = self.get_clock().now()
                self.get_logger().info(
                    f"ASR turn started: session={new_session_id} "
                    f"user_turn_id={new_user_turn_id}"
                )
                self._emit_event(
                    AsrEvent.EVENT_STARTUP_IDLE,
                    "turn_context_active_waiting",
                    transition_to_state=AsrState.STATE_WAITING,
                    session_id=new_session_id,
                    user_turn_id=new_user_turn_id,
                    source_id=self.expected_source_id,
                    stream_id=self.expected_stream_id,
                    context_active=True,
                )

            self._active_session_id = new_session_id
            self._active_user_turn_id = new_user_turn_id
            self._context_active = True

    def on_control_event(self, event: ControlEvent) -> None:
        with self._turn_state_lock:
            config = self._control_config_for_event(event)
            window = self._control_windows[event.control_id]
            self._emit_event(
                AsrEvent.EVENT_CONTROL_RECEIVED,
                "control_received",
                control_id=event.control_id,
                source_id=event.source_id,
                stream_id=event.stream_id,
                session_id=self._active_session_id,
                user_turn_id=self._active_user_turn_id,
                context_active=self._context_active,
                window_open=window.open,
                start_unix_ns=event.stamp_unix_ns,
                end_unix_ns=event.stamp_unix_ns,
            )
            if event.source_id != config.source_id or event.stream_id != config.stream_id:
                self.get_logger().error(
                    f"Dropping control event {event.control_id}: source_id/stream_id mismatch"
                )
                self._emit_event(
                    AsrEvent.EVENT_CONTROL_REJECTED_IDENTITY,
                    "control_rejected_source_stream_mismatch",
                    control_id=event.control_id,
                    source_id=event.source_id,
                    stream_id=event.stream_id,
                    session_id=self._active_session_id,
                    user_turn_id=self._active_user_turn_id,
                    error_code="source_stream_mismatch",
                )
                return
            if event.active:
                self._last_speech = self.get_clock().now()

            should_open = event.start or (event.active and not window.previous_active)
            should_close = event.end or (window.previous_active and not event.active)

            if should_open:
                self._open_control_window(config, window, event)
            if should_close:
                self._close_control_window(config, window, event)

            window.previous_active = event.active

    def _control_config_for_event(self, event: ControlEvent) -> ControlInputConfig:
        for config in self.control_configs:
            if config.control_id == event.control_id:
                return config
        raise RuntimeError(f"unknown control event ID: {event.control_id}")

    def _open_control_window(
        self,
        config: ControlInputConfig,
        window: ControlWindowState,
        event: ControlEvent,
    ) -> None:
        pre_roll_ns = self._milliseconds_to_nanoseconds(config.pre_roll_ms)
        start_unix_ns = event.stamp_unix_ns - pre_roll_ns
        if start_unix_ns <= 0:
            self.get_logger().error(
                f"Dropping control window {event.control_id}: pre-roll starts before valid unix time"
            )
            self._emit_event(
                AsrEvent.EVENT_INVALID_CLOSE_TIME,
                "control_window_invalid_open_time",
                control_id=event.control_id,
                source_id=event.source_id,
                stream_id=event.stream_id,
                session_id=self._active_session_id,
                user_turn_id=self._active_user_turn_id,
                start_unix_ns=start_unix_ns,
                error_code=ERROR_TIME_RANGE_UNRESOLVED,
            )
            window.open = False
            window.start_unix_ns = 0
            return
        stream_session: AsrStreamingSession | None = None
        if self.input_capability.streaming:
            if not self._context_active or not self._active_session_id:
                self._emit_event(
                    AsrEvent.EVENT_SUBMIT_SKIPPED_CONTEXT_INACTIVE,
                    "stream_open_skipped_context_inactive",
                    control_id=event.control_id,
                    source_id=event.source_id,
                    stream_id=event.stream_id,
                    start_unix_ns=start_unix_ns,
                )
                window.open = False
                window.start_unix_ns = 0
                return
            try:
                with self._backend_lock:
                    stream_session = self._streaming_backend().start_stream(
                        AsrStreamRequest(
                            session_id=self._active_session_id,
                            user_turn_id=self._active_user_turn_id,
                        )
                    )
            except Exception as exc:
                self._handle_stream_error(
                    "stream_open_failed",
                    exc,
                    control_id=event.control_id,
                    session_id=self._active_session_id,
                    user_turn_id=self._active_user_turn_id,
                    sample_count=0,
                    start_unix_ns=start_unix_ns,
                    end_unix_ns=start_unix_ns,
                )
                return
        window.open = True
        window.start_unix_ns = start_unix_ns
        window.stream_session = stream_session
        window.stream_session_id = self._active_session_id if stream_session is not None else ""
        window.stream_user_turn_id = self._active_user_turn_id if stream_session is not None else 0
        window.stream_sample_count = 0
        self._emit_event(
            AsrEvent.EVENT_CONTROL_WINDOW_OPENED,
            "control_window_opened",
            transition_to_state=AsrState.STATE_COLLECTING,
            control_id=event.control_id,
            source_id=event.source_id,
            stream_id=event.stream_id,
            session_id=self._active_session_id,
            user_turn_id=self._active_user_turn_id,
            context_active=self._context_active,
            window_open=True,
            start_unix_ns=start_unix_ns,
        )
        if stream_session is not None:
            self._emit_event(
                AsrEvent.EVENT_STREAM_OPENED,
                "stream_opened",
                control_id=event.control_id,
                source_id=event.source_id,
                stream_id=event.stream_id,
                session_id=window.stream_session_id,
                user_turn_id=window.stream_user_turn_id,
                context_active=self._context_active,
                window_open=True,
                start_unix_ns=start_unix_ns,
            )

    def _close_control_window(
        self,
        config: ControlInputConfig,
        window: ControlWindowState,
        event: ControlEvent,
    ) -> None:
        if not window.open:
            self._emit_event(
                AsrEvent.EVENT_CLOSE_IGNORED_NO_WINDOW,
                "control_close_ignored_no_window_open",
                control_id=event.control_id,
                source_id=event.source_id,
                stream_id=event.stream_id,
                session_id=self._active_session_id,
                user_turn_id=self._active_user_turn_id,
                context_active=self._context_active,
                start_unix_ns=event.stamp_unix_ns,
                end_unix_ns=event.stamp_unix_ns,
            )
            return
        post_roll_ns = self._milliseconds_to_nanoseconds(config.post_roll_ms)
        end_unix_ns = event.stamp_unix_ns + post_roll_ns
        start_unix_ns = window.start_unix_ns
        session = window.stream_session
        session_id = window.stream_session_id
        user_turn_id = window.stream_user_turn_id
        sample_count = window.stream_sample_count
        window.open = False
        self._emit_event(
            AsrEvent.EVENT_CONTROL_WINDOW_CLOSED,
            "control_window_closed",
            transition_to_state=AsrState.STATE_WAITING,
            control_id=event.control_id,
            source_id=event.source_id,
            stream_id=event.stream_id,
            session_id=session_id if session is not None else self._active_session_id,
            user_turn_id=user_turn_id if session is not None else self._active_user_turn_id,
            context_active=self._context_active,
            start_unix_ns=start_unix_ns,
            end_unix_ns=end_unix_ns,
        )
        if self.input_capability.streaming:
            self._finish_streaming_window(
                control_id=event.control_id,
                session=session,
                session_id=session_id,
                user_turn_id=user_turn_id,
                sample_count=sample_count,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
                window=window,
            )
            return
        window.start_unix_ns = 0
        if not config.submit_on_close:
            self._emit_event(
                AsrEvent.EVENT_SUBMIT_SKIPPED_SUBMIT_ON_CLOSE_FALSE,
                "submit_skipped_submit_on_close_false",
                control_id=event.control_id,
                source_id=event.source_id,
                stream_id=event.stream_id,
                session_id=self._active_session_id,
                user_turn_id=self._active_user_turn_id,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
            )
            return
        if not self._context_active or not self._active_session_id:
            self._emit_event(
                AsrEvent.EVENT_SUBMIT_SKIPPED_CONTEXT_INACTIVE,
                "submit_skipped_context_inactive",
                control_id=event.control_id,
                source_id=event.source_id,
                stream_id=event.stream_id,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
            )
            return
        if end_unix_ns <= start_unix_ns:
            self.get_logger().error(
                f"Dropping control window {event.control_id}: close time must be after open time"
            )
            self._emit_event(
                AsrEvent.EVENT_INVALID_CLOSE_TIME,
                "control_window_invalid_close_time",
                control_id=event.control_id,
                source_id=event.source_id,
                stream_id=event.stream_id,
                session_id=self._active_session_id,
                user_turn_id=self._active_user_turn_id,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
                error_code=ERROR_TIME_RANGE_UNRESOLVED,
            )
            return
        self._submit_timeline_window(
            start_unix_ns=start_unix_ns,
            end_unix_ns=end_unix_ns,
            reason=f"control:{event.control_id}:close",
        )

    def _finish_streaming_window(
        self,
        *,
        control_id: str,
        session: AsrStreamingSession | None,
        session_id: str,
        user_turn_id: int,
        sample_count: int,
        start_unix_ns: int,
        end_unix_ns: int,
        window: ControlWindowState,
    ) -> None:
        if session is None:
            self._handle_stream_error(
                "stream_close_missing_session",
                RuntimeError("streaming control window has no backend stream session"),
                control_id=control_id,
                session_id=session_id,
                user_turn_id=user_turn_id,
                sample_count=sample_count,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
            )
            self._reset_streaming_window(window)
            return

        try:
            with self._backend_lock:
                pending_results = tuple(session.drain_results())
                final_results = tuple(session.finish())
                finished_results = tuple(session.drain_results())
        except Exception as exc:
            self._handle_stream_error(
                "stream_close_failed",
                exc,
                control_id=control_id,
                session_id=session_id,
                user_turn_id=user_turn_id,
                sample_count=sample_count,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
            )
            self._reset_streaming_window(window)
            return

        final_published = self._publish_stream_results(
            control_id=control_id,
            session_id=session_id,
            user_turn_id=user_turn_id,
            results=pending_results,
            sample_count=sample_count,
            start_unix_ns=start_unix_ns,
            end_unix_ns=end_unix_ns,
        )
        final_published = (
            self._publish_stream_results(
                control_id=control_id,
                session_id=session_id,
                user_turn_id=user_turn_id,
                results=final_results,
                sample_count=sample_count,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
            )
            or final_published
        )
        final_published = (
            self._publish_stream_results(
                control_id=control_id,
                session_id=session_id,
                user_turn_id=user_turn_id,
                results=finished_results,
                sample_count=sample_count,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
            )
            or final_published
        )
        self._emit_event(
            AsrEvent.EVENT_STREAM_CLOSED,
            "stream_closed",
            control_id=control_id,
            source_id=self.expected_source_id,
            stream_id=self.expected_stream_id,
            session_id=session_id,
            user_turn_id=user_turn_id,
            context_active=self._context_active,
            sample_count=sample_count,
            start_unix_ns=start_unix_ns,
            end_unix_ns=end_unix_ns,
        )
        self._reset_streaming_window(window)
        if not final_published:
            self._handle_stream_error(
                "stream_final_missing",
                RuntimeError("streaming ASR backend closed without a final result"),
                control_id=control_id,
                session_id=session_id,
                user_turn_id=user_turn_id,
                sample_count=sample_count,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
            )

    def _push_audio_to_open_streams(
        self,
        payload: AsrAudioPayload,
        *,
        start_unix_ns: int,
    ) -> None:
        if not self.input_capability.streaming:
            return
        for control_id, window in self._control_windows.items():
            session = window.stream_session
            if not window.open or session is None:
                continue
            try:
                with self._backend_lock:
                    push_results = tuple(session.push_audio(payload))
                    drained_results = tuple(session.drain_results())
                window.stream_sample_count += payload.sample_count
                self._emit_event(
                    AsrEvent.EVENT_STREAM_AUDIO_PUSHED,
                    "stream_audio_pushed",
                    control_id=control_id,
                    source_id=self.expected_source_id,
                    stream_id=self.expected_stream_id,
                    session_id=window.stream_session_id,
                    user_turn_id=window.stream_user_turn_id,
                    context_active=self._context_active,
                    window_open=True,
                    sample_count=payload.sample_count,
                    start_unix_ns=start_unix_ns,
                )
                self._publish_stream_results(
                    control_id=control_id,
                    session_id=window.stream_session_id,
                    user_turn_id=window.stream_user_turn_id,
                    results=push_results + drained_results,
                    sample_count=window.stream_sample_count,
                    start_unix_ns=window.start_unix_ns,
                    end_unix_ns=start_unix_ns,
                )
            except Exception as exc:
                self._handle_stream_error(
                    "stream_audio_push_failed",
                    exc,
                    control_id=control_id,
                    session_id=window.stream_session_id,
                    user_turn_id=window.stream_user_turn_id,
                    sample_count=window.stream_sample_count,
                    start_unix_ns=window.start_unix_ns,
                    end_unix_ns=start_unix_ns,
                )
                self._reset_streaming_window(window)

    def _publish_stream_results(
        self,
        *,
        control_id: str,
        session_id: str,
        user_turn_id: int,
        results: Iterable[AsrStreamResult],
        sample_count: int,
        start_unix_ns: int,
        end_unix_ns: int,
    ) -> bool:
        final_published = False
        for result in results:
            if result.sample_count > sample_count:
                raise RuntimeError(
                    "ASR stream result sample_count exceeds pushed audio sample count"
                )
            validated_transcript = build_asr_transcript(
                result.transcript.segments,
                sample_count=result.sample_count,
            )
            if result.is_final:
                final_published = True
                self._publish_result(
                    session_id,
                    user_turn_id,
                    AsrResult.STATUS_FINAL,
                    "stream_final",
                    asr_transcript_text(validated_transcript),
                )
                self._emit_event(
                    AsrEvent.EVENT_STREAM_FINAL_RESULT,
                    "stream_final_result",
                    transition_to_state=AsrState.STATE_COMPLETED,
                    control_id=control_id,
                    source_id=self.expected_source_id,
                    stream_id=self.expected_stream_id,
                    session_id=session_id,
                    user_turn_id=user_turn_id,
                    context_active=self._context_active,
                    sample_count=result.sample_count,
                    start_unix_ns=start_unix_ns,
                    end_unix_ns=end_unix_ns,
                )
                continue
            self._publish_result(
                session_id,
                user_turn_id,
                AsrResult.STATUS_PARTIAL,
                "stream_partial",
                asr_transcript_text(validated_transcript),
            )
            self._emit_event(
                AsrEvent.EVENT_STREAM_PARTIAL_RESULT,
                "stream_partial_result",
                transition_to_state=AsrState.STATE_TRANSCRIBING,
                control_id=control_id,
                source_id=self.expected_source_id,
                stream_id=self.expected_stream_id,
                session_id=session_id,
                user_turn_id=user_turn_id,
                context_active=self._context_active,
                window_open=True,
                sample_count=result.sample_count,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
            )
        return final_published

    def _handle_stream_error(
        self,
        reason: str,
        exc: Exception,
        *,
        control_id: str,
        session_id: str,
        user_turn_id: int,
        sample_count: int,
        start_unix_ns: int,
        end_unix_ns: int,
    ) -> None:
        error_text = str(exc)
        self.get_logger().error(f"ASR stream failed: {error_text}")
        if session_id:
            self._publish_result(
                session_id,
                user_turn_id,
                AsrResult.STATUS_ERROR,
                reason,
                "",
            )
        self._emit_event(
            AsrEvent.EVENT_STREAM_ERROR,
            reason,
            transition_to_state=AsrState.STATE_FAILED,
            control_id=control_id,
            source_id=self.expected_source_id,
            stream_id=self.expected_stream_id,
            session_id=session_id,
            user_turn_id=user_turn_id,
            context_active=self._context_active,
            sample_count=sample_count,
            start_unix_ns=start_unix_ns,
            end_unix_ns=end_unix_ns,
            error_code=error_text,
        )
        self._fail_closed(f"ASR stream failed: {error_text}")

    def _cancel_open_streams(self, reason: str) -> None:
        for control_id, window in self._control_windows.items():
            session = window.stream_session
            if session is None:
                continue
            session_id = window.stream_session_id
            user_turn_id = window.stream_user_turn_id
            sample_count = window.stream_sample_count
            start_unix_ns = window.start_unix_ns
            try:
                with self._backend_lock:
                    session.cancel()
            except Exception as exc:
                self._handle_stream_error(
                    "stream_cancel_failed",
                    exc,
                    control_id=control_id,
                    session_id=session_id,
                    user_turn_id=user_turn_id,
                    sample_count=sample_count,
                    start_unix_ns=start_unix_ns,
                    end_unix_ns=start_unix_ns,
                )
                self._reset_streaming_window(window)
                continue
            self._emit_event(
                AsrEvent.EVENT_STREAM_CANCELLED,
                reason,
                control_id=control_id,
                source_id=self.expected_source_id,
                stream_id=self.expected_stream_id,
                session_id=session_id,
                user_turn_id=user_turn_id,
                context_active=self._context_active,
                sample_count=sample_count,
                start_unix_ns=start_unix_ns,
                end_unix_ns=start_unix_ns,
            )
            self._reset_streaming_window(window)

    @staticmethod
    def _reset_streaming_window(window: ControlWindowState) -> None:
        window.open = False
        window.start_unix_ns = 0
        window.stream_session = None
        window.stream_session_id = ""
        window.stream_user_turn_id = 0
        window.stream_sample_count = 0

    def _submit_timeline_window(
        self,
        *,
        start_unix_ns: int,
        end_unix_ns: int,
        reason: str,
    ) -> None:
        try:
            with self._timeline_lock:
                timeline_slice = self._timeline.slice(
                    NumericTimeRange(
                        start_unix_ns=start_unix_ns,
                        end_unix_ns=end_unix_ns,
                    )
                )
        except TimelineRangeError as exc:
            self.get_logger().error(f"Dropping ASR control window: {exc}")
            self._emit_timeline_slice_failure(
                exc,
                reason=reason,
                session_id=self._active_session_id,
                user_turn_id=self._active_user_turn_id,
                start_unix_ns=start_unix_ns,
                end_unix_ns=end_unix_ns,
            )
            return

        duration_sec = float(timeline_slice.sample_count) / float(timeline_slice.sample_rate)
        if duration_sec < self.min_audio_sec:
            self.get_logger().debug(
                f"ASR control window too short for {reason}: {duration_sec:.3f}s"
            )
            self._emit_event(
                AsrEvent.EVENT_WINDOW_TOO_SHORT,
                "window_too_short",
                session_id=self._active_session_id,
                user_turn_id=self._active_user_turn_id,
                source_id=self.expected_source_id,
                stream_id=self.expected_stream_id,
                sample_count=timeline_slice.sample_count,
                start_unix_ns=timeline_slice.time_range.start_unix_ns,
                end_unix_ns=timeline_slice.time_range.end_unix_ns,
            )
            return

        self._jobs.put(
            TranscriptionJob(
                session_id=self._active_session_id,
                user_turn_id=self._active_user_turn_id,
                payload=timeline_slice.payload,
                reason=reason,
            )
        )
        self._emit_event(
            AsrEvent.EVENT_JOB_QUEUED,
            "job_queued",
            transition_to_state=AsrState.STATE_QUEUED,
            session_id=self._active_session_id,
            user_turn_id=self._active_user_turn_id,
            source_id=self.expected_source_id,
            stream_id=self.expected_stream_id,
            sample_count=timeline_slice.sample_count,
            start_unix_ns=timeline_slice.time_range.start_unix_ns,
            end_unix_ns=timeline_slice.time_range.end_unix_ns,
        )

    def _emit_timeline_slice_failure(
        self,
        exc: TimelineRangeError,
        *,
        reason: str,
        session_id: str,
        user_turn_id: int,
        start_unix_ns: int,
        end_unix_ns: int,
    ) -> None:
        event = AsrEvent.EVENT_TIMELINE_SLICE_UNAVAILABLE
        if exc.error_code == ERROR_RANGE_NOT_CONTINUOUS:
            event = AsrEvent.EVENT_TIMELINE_SLICE_NOT_CONTINUOUS
        elif "overlap" in str(exc) or "overlapping" in str(exc):
            event = AsrEvent.EVENT_TIMELINE_OVERLAP_DERIVED_FAILURE
        self._emit_event(
            event,
            reason,
            session_id=session_id,
            user_turn_id=user_turn_id,
            source_id=self.expected_source_id,
            stream_id=self.expected_stream_id,
            start_unix_ns=start_unix_ns,
            end_unix_ns=end_unix_ns,
            error_code=exc.error_code,
        )

    @staticmethod
    def _milliseconds_to_nanoseconds(milliseconds: float) -> int:
        return int(round(milliseconds * 1_000_000.0))

    def _vad_state_to_control_event(
        self,
        config: ControlInputConfig,
        msg: VadState,
    ) -> ControlEvent:
        return ControlEvent(
            control_id=config.control_id,
            source_id=msg.source_id,
            stream_id=msg.stream_id,
            active=msg.is_speech,
            start=msg.start,
            end=msg.end,
            stamp_unix_ns=self._stamp_to_unix_ns(msg.header.stamp),
        )

    def on_audio(self, msg: AudioFrame) -> None:
        try:
            if int(msg.sample_rate) != self.target_sample_rate:
                raise ValueError(
                    "AudioFrame sample_rate must match target_sample_rate "
                    f"{self.target_sample_rate}, got {msg.sample_rate}"
                )
            payload = self._frame_to_payload(
                msg,
                expected_source_id=self.expected_source_id,
                expected_stream_id=self.expected_stream_id,
                expected_encoding=self.input_capability.audio_encoding,
                expected_channels=self.input_capability.channels,
            )
        except ValueError as exc:
            self.get_logger().error(f"Dropping invalid AudioFrame: {exc}")
            self._emit_event(
                AsrEvent.EVENT_INVALID_AUDIO_FRAME_DROPPED,
                "invalid_audio_frame_dropped",
                source_id=str(msg.source_id),
                stream_id=str(msg.stream_id),
                error_code=str(exc),
            )
            return

        try:
            start_unix_ns = self._stamp_to_unix_ns(msg.header.stamp)
            with self._timeline_lock:
                self._timeline.append(start_unix_ns=start_unix_ns, payload=payload)
        except TimelineRangeError as exc:
            self.get_logger().error(f"Dropping AudioFrame from ASR timeline: {exc}")
            event = AsrEvent.EVENT_AUDIO_FRAME_TIMELINE_APPEND_DROPPED
            if "overlap" in str(exc) or "overlapping" in str(exc):
                event = AsrEvent.EVENT_TIMELINE_OVERLAP_DERIVED_FAILURE
            self._emit_event(
                event,
                "audio_frame_timeline_append_dropped",
                source_id=msg.source_id,
                stream_id=msg.stream_id,
                sample_count=payload.sample_count,
                error_code=exc.error_code,
            )
            return

        with self._turn_state_lock:
            self._push_audio_to_open_streams(payload, start_unix_ns=start_unix_ns)
            if self.input_capability.streaming:
                return
            if not self.control_default_enabled:
                return
            if not self._context_active or not self._active_session_id:
                return
            with self._samples_lock:
                self._payload_chunks.append(payload.data)
                self._buffer_sample_count += payload.sample_count

    def handle_transcribe_audio(
        self,
        request: TranscribeAudio.Request,
        response: TranscribeAudio.Response,
    ) -> TranscribeAudio.Response:
        audio_scope = request.audio_scope.strip()
        if audio_scope and audio_scope != self.expected_stream_id:
            return self._transcribe_error(
                response,
                TranscribeAudio.Response.ERROR_UNSUPPORTED_AUDIO_SCOPE,
                "audio_scope must be empty or match expected_stream_id",
            )

        try:
            time_range = parse_numeric_time_range(request.time_range_spec)
            with self._timeline_lock:
                timeline_slice = self._timeline.slice(time_range)
        except TimelineRangeError as exc:
            self._emit_timeline_slice_failure(
                exc,
                reason="transcribe_audio_service_slice_failed",
                session_id="timeline",
                user_turn_id=0,
                start_unix_ns=0,
                end_unix_ns=0,
            )
            return self._transcribe_error(response, exc.error_code, str(exc))

        duration_sec = float(timeline_slice.sample_count) / float(timeline_slice.sample_rate)
        if duration_sec < self.min_audio_sec:
            self._emit_event(
                AsrEvent.EVENT_WINDOW_TOO_SHORT,
                "requested_audio_too_short",
                session_id="timeline",
                user_turn_id=0,
                source_id=self.expected_source_id,
                stream_id=self.expected_stream_id,
                sample_count=timeline_slice.sample_count,
                start_unix_ns=timeline_slice.time_range.start_unix_ns,
                end_unix_ns=timeline_slice.time_range.end_unix_ns,
            )
            return self._transcribe_error(
                response,
                TranscribeAudio.Response.ERROR_TRANSCRIBE_FAILED,
                "requested audio duration is below min_audio_sec",
            )

        try:
            self._emit_event(
                AsrEvent.EVENT_BACKEND_TRANSCRIPTION_STARTED,
                "backend_transcription_started",
                transition_to_state=AsrState.STATE_TRANSCRIBING,
                session_id="timeline",
                user_turn_id=0,
                source_id=self.expected_source_id,
                stream_id=self.expected_stream_id,
                sample_count=timeline_slice.sample_count,
                start_unix_ns=timeline_slice.time_range.start_unix_ns,
                end_unix_ns=timeline_slice.time_range.end_unix_ns,
            )
            with self._backend_lock:
                transcript = self.backend.transcribe(
                    AsrRequest(
                        session_id="timeline",
                        user_turn_id=0,
                        payload=timeline_slice.payload,
                    )
                )
            self._fill_transcribe_success(response, timeline_slice, transcript)
            self._emit_event(
                AsrEvent.EVENT_BACKEND_COMPLETED,
                "backend_completed",
                transition_to_state=AsrState.STATE_COMPLETED,
                session_id="timeline",
                user_turn_id=0,
                source_id=self.expected_source_id,
                stream_id=self.expected_stream_id,
                sample_count=timeline_slice.sample_count,
                start_unix_ns=timeline_slice.time_range.start_unix_ns,
                end_unix_ns=timeline_slice.time_range.end_unix_ns,
            )
        except TimeoutError:
            self._emit_event(
                AsrEvent.EVENT_BACKEND_TIMEOUT,
                "backend_timeout",
                transition_to_state=AsrState.STATE_TIMEOUT,
                session_id="timeline",
                user_turn_id=0,
                source_id=self.expected_source_id,
                stream_id=self.expected_stream_id,
                sample_count=timeline_slice.sample_count,
                error_code=TranscribeAudio.Response.ERROR_TRANSCRIBE_FAILED,
            )
            return self._transcribe_error(
                response,
                TranscribeAudio.Response.ERROR_TRANSCRIBE_FAILED,
                "ASR backend timed out",
            )
        except Exception as exc:
            self.get_logger().error(f"ASR timeline transcription failed: {exc}")
            self._emit_event(
                AsrEvent.EVENT_BACKEND_ERROR,
                "backend_error",
                transition_to_state=AsrState.STATE_FAILED,
                session_id="timeline",
                user_turn_id=0,
                source_id=self.expected_source_id,
                stream_id=self.expected_stream_id,
                sample_count=timeline_slice.sample_count,
                error_code=TranscribeAudio.Response.ERROR_TRANSCRIBE_FAILED,
            )
            return self._transcribe_error(
                response,
                TranscribeAudio.Response.ERROR_TRANSCRIBE_FAILED,
                f"ASR backend failed: {exc}",
            )

        return response

    def _transcribe_error(
        self,
        response: TranscribeAudio.Response,
        error_code: str,
        message: str,
    ) -> TranscribeAudio.Response:
        response.success = False
        response.error_code = error_code
        response.message = message
        response.segments = []
        response.time_range = ResolvedTimeRange()
        response.audio_window_ref = AudioWindowRef()
        response.model_ref = AudioModelRef()
        return response

    def _fill_transcribe_success(
        self,
        response: TranscribeAudio.Response,
        timeline_slice: TimelineSlice,
        transcript: AsrTranscript,
    ) -> None:
        validated_transcript = build_asr_transcript(
            transcript.segments,
            sample_count=timeline_slice.sample_count,
        )
        time_range_msg = self._resolved_time_range_msg(timeline_slice.time_range)
        response.success = True
        response.error_code = TranscribeAudio.Response.ERROR_NONE
        response.message = ""
        response.time_range = time_range_msg
        response.audio_window_ref = AudioWindowRef(
            window_id=self.timeline_window_id,
            window_epoch=self.timeline_window_epoch,
            source_id=self.expected_source_id,
            stream_id=self.expected_stream_id,
            time_range=time_range_msg,
        )
        response.model_ref = AudioModelRef(
            backend_name=self.backend_name,
            backend_kind=self.backend_kind,
            model_id=self.backend_model,
            model_path=self.backend_model_path,
            model_version=self.backend_model_version,
            model_revision=self.backend_model_revision,
        )
        response.segments = self._transcript_segment_msgs(
            timeline_slice,
            validated_transcript,
        )

    def _transcript_segment_msgs(
        self,
        timeline_slice: TimelineSlice,
        transcript: AsrTranscript,
    ) -> list[TranscriptSegment]:
        messages: list[TranscriptSegment] = []
        for segment in transcript.segments:
            messages.append(
                TranscriptSegment(
                    start_unix_ns=self._relative_sample_start_unix_ns(
                        timeline_slice,
                        segment.start_sample,
                    ),
                    end_unix_ns=self._relative_sample_end_unix_ns(
                        timeline_slice,
                        segment.end_sample,
                    ),
                    text=segment.text,
                    speaker_label=segment.speaker_label if segment.speaker_label else "",
                )
            )
        return messages

    @staticmethod
    def _relative_sample_start_unix_ns(
        timeline_slice: TimelineSlice,
        sample_index: int,
    ) -> int:
        duration_ns = sample_index * _NSEC_PER_SEC // timeline_slice.sample_rate
        return timeline_slice.time_range.start_unix_ns + duration_ns

    @staticmethod
    def _relative_sample_end_unix_ns(
        timeline_slice: TimelineSlice,
        sample_index: int,
    ) -> int:
        numerator = sample_index * _NSEC_PER_SEC
        duration_ns, remainder = divmod(numerator, timeline_slice.sample_rate)
        if remainder != 0:
            duration_ns += 1
        return timeline_slice.time_range.start_unix_ns + duration_ns

    def _resolved_time_range_msg(self, time_range: NumericTimeRange) -> ResolvedTimeRange:
        return ResolvedTimeRange(
            start_unix_ns=time_range.start_unix_ns,
            end_unix_ns=time_range.end_unix_ns,
            clock=self.timeline_clock,
            uncertainty_ns=0,
            uncertainty_reason="",
        )

    def _check_timeout(self) -> None:
        with self._turn_state_lock:
            if self.input_capability.streaming:
                return
            if not self.control_default_enabled:
                return
            if not self._context_active:
                return
            now = self.get_clock().now()
            elapsed = (now - self._last_speech).nanoseconds / 1e9
            if elapsed < self.silence_timeout_sec:
                return
            self.get_logger().info(f"ASR silence timeout: {elapsed:.1f}s")
            self._submit_current_buffer("silence_timeout", publish_timeout_if_empty=True)
            self._last_speech = now

    def _submit_current_buffer(self, reason: str, *, publish_timeout_if_empty: bool) -> None:
        with self._turn_state_lock:
            if not self._active_session_id:
                self._emit_event(
                    AsrEvent.EVENT_SUBMIT_SKIPPED_CONTEXT_INACTIVE,
                    "submit_skipped_context_inactive",
                    source_id=self.expected_source_id,
                    stream_id=self.expected_stream_id,
                )
                return
            session_id = self._active_session_id
            user_turn_id = self._active_user_turn_id
            with self._samples_lock:
                data = b"".join(self._payload_chunks)
                sample_count = self._buffer_sample_count
                self._payload_chunks.clear()
                self._buffer_sample_count = 0

        duration_sec = float(sample_count) / float(self.target_sample_rate)
        if duration_sec < self.min_audio_sec:
            if publish_timeout_if_empty:
                self._publish_result(
                    session_id,
                    user_turn_id,
                    AsrResult.STATUS_TIMEOUT,
                    reason,
                    "",
                )
            else:
                self.get_logger().debug(
                    f"ASR buffer too short for {reason}: {duration_sec:.3f}s"
                )
            self._emit_event(
                AsrEvent.EVENT_WINDOW_TOO_SHORT,
                "buffer_too_short",
                transition_to_state=(
                    AsrState.STATE_TIMEOUT if publish_timeout_if_empty else None
                ),
                session_id=session_id,
                user_turn_id=user_turn_id,
                source_id=self.expected_source_id,
                stream_id=self.expected_stream_id,
                sample_count=sample_count,
            )
            return

        self._jobs.put(
            TranscriptionJob(
                session_id=session_id,
                user_turn_id=user_turn_id,
                payload=AsrAudioPayload(
                    encoding=self.input_capability.audio_encoding,
                    sample_rate_hz=self.target_sample_rate,
                    channels=self.input_capability.channels,
                    data=data,
                    sample_count=sample_count,
                ),
                reason=reason,
            )
        )
        self._emit_event(
            AsrEvent.EVENT_JOB_QUEUED,
            "job_queued",
            transition_to_state=AsrState.STATE_QUEUED,
            session_id=session_id,
            user_turn_id=user_turn_id,
            source_id=self.expected_source_id,
            stream_id=self.expected_stream_id,
            sample_count=sample_count,
        )

    def _clear_buffer(self) -> None:
        with self._samples_lock:
            self._payload_chunks.clear()
            self._buffer_sample_count = 0

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._run_transcription(job)

    def _run_transcription(self, job: TranscriptionJob) -> None:
        try:
            self._emit_event(
                AsrEvent.EVENT_BACKEND_TRANSCRIPTION_STARTED,
                "backend_transcription_started",
                transition_to_state=AsrState.STATE_TRANSCRIBING,
                session_id=job.session_id,
                user_turn_id=job.user_turn_id,
                source_id=getattr(self, "expected_source_id", ""),
                stream_id=getattr(self, "expected_stream_id", ""),
                sample_count=job.payload.sample_count,
            )
            with self._backend_lock:
                transcript = self.backend.transcribe(
                    AsrRequest(
                        session_id=job.session_id,
                        user_turn_id=job.user_turn_id,
                        payload=job.payload,
                    )
                )
            validated_transcript = build_asr_transcript(
                transcript.segments,
                sample_count=job.payload.sample_count,
            )
            self._publish_result(
                job.session_id,
                job.user_turn_id,
                AsrResult.STATUS_FINAL,
                job.reason,
                asr_transcript_text(validated_transcript),
            )
            self._emit_event(
                AsrEvent.EVENT_BACKEND_COMPLETED,
                "backend_completed",
                transition_to_state=AsrState.STATE_COMPLETED,
                session_id=job.session_id,
                user_turn_id=job.user_turn_id,
                source_id=getattr(self, "expected_source_id", ""),
                stream_id=getattr(self, "expected_stream_id", ""),
                sample_count=job.payload.sample_count,
            )
        except TimeoutError:
            self._publish_result(
                job.session_id,
                job.user_turn_id,
                AsrResult.STATUS_ERROR,
                "backend_timeout",
                "",
            )
            self._emit_event(
                AsrEvent.EVENT_BACKEND_TIMEOUT,
                "backend_timeout",
                transition_to_state=AsrState.STATE_TIMEOUT,
                session_id=job.session_id,
                user_turn_id=job.user_turn_id,
                source_id=getattr(self, "expected_source_id", ""),
                stream_id=getattr(self, "expected_stream_id", ""),
                sample_count=job.payload.sample_count,
            )
            self._fail_closed("ASR backend timed out")
        except Exception as exc:
            self.get_logger().error(f"ASR transcription failed: {exc}")
            self._publish_result(
                job.session_id,
                job.user_turn_id,
                AsrResult.STATUS_ERROR,
                "backend_error",
                "",
            )
            self._emit_event(
                AsrEvent.EVENT_BACKEND_ERROR,
                "backend_error",
                transition_to_state=AsrState.STATE_FAILED,
                session_id=job.session_id,
                user_turn_id=job.user_turn_id,
                source_id=getattr(self, "expected_source_id", ""),
                stream_id=getattr(self, "expected_stream_id", ""),
                sample_count=job.payload.sample_count,
            )
            self._fail_closed(f"ASR backend failed: {exc}")

    def _fail_closed(self, reason: str) -> None:
        if getattr(self, "_fail_closed_triggered", False):
            return
        self._fail_closed_triggered = True
        self._emit_event(
            AsrEvent.EVENT_FAIL_CLOSED,
            "fail_closed",
            transition_to_state=AsrState.STATE_FAILED,
            session_id=getattr(self, "_active_session_id", ""),
            user_turn_id=getattr(self, "_active_user_turn_id", 0),
            source_id=getattr(self, "expected_source_id", ""),
            stream_id=getattr(self, "expected_stream_id", ""),
            error_code=reason,
        )
        self.get_logger().fatal(f"ASR fail closed: {reason}")
        rclpy.shutdown()

    @staticmethod
    def _frame_to_payload(
        msg: AudioFrame,
        *,
        expected_source_id: str,
        expected_stream_id: str,
        expected_encoding: str,
        expected_channels: int,
    ) -> AsrAudioPayload:
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
        if int(msg.channels) != expected_channels:
            raise ValueError(f"AudioFrame channels must be {expected_channels}, got {msg.channels}")
        if msg.encoding != expected_encoding:
            raise ValueError(
                f"AudioFrame encoding must be {expected_encoding}, got {msg.encoding}"
            )
        if int(msg.sample_rate) <= 0:
            raise ValueError(f"AudioFrame sample_rate must be positive, got {msg.sample_rate}")
        data = bytes(msg.data)
        if expected_encoding == "FLOAT32LE":
            if int(msg.bit_depth) != 32:
                raise ValueError(f"AudioFrame bit_depth must be 32, got {msg.bit_depth}")
            if len(data) % np.dtype("<f4").itemsize != 0:
                raise ValueError("AudioFrame float32 data length is not byte-aligned")
            samples = np.frombuffer(data, dtype="<f4")
            return AsrAudioPayload.from_float32_samples(
                samples,
                sample_rate_hz=int(msg.sample_rate),
                channels=expected_channels,
            )
        if expected_encoding == "PCM16LE":
            if int(msg.bit_depth) != 16:
                raise ValueError(f"AudioFrame bit_depth must be 16, got {msg.bit_depth}")
            return AsrAudioPayload.from_pcm16le_bytes(
                data,
                sample_rate_hz=int(msg.sample_rate),
                channels=expected_channels,
            )
        raise ValueError(f"Unsupported ASR input encoding: {expected_encoding}")

    @staticmethod
    def _stamp_to_unix_ns(stamp: Time) -> int:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
        if sec < 0:
            raise TimelineRangeError(
                ERROR_TIME_RANGE_UNRESOLVED,
                "AudioFrame header.stamp.sec must be non-negative",
            )
        if nanosec < 0 or nanosec >= 1_000_000_000:
            raise TimelineRangeError(
                ERROR_TIME_RANGE_UNRESOLVED,
                "AudioFrame header.stamp.nanosec must be within [0, 1000000000)",
            )
        unix_ns = sec * 1_000_000_000 + nanosec
        if unix_ns <= 0:
            raise TimelineRangeError(
                ERROR_TIME_RANGE_UNRESOLVED,
                "AudioFrame header.stamp must not be zero",
            )
        return unix_ns

    def _publish_result(
        self,
        session_id: str,
        user_turn_id: int,
        status: int,
        reason: str,
        text: str,
    ) -> None:
        msg = AsrResult()
        msg.timestamp = self.get_clock().now().to_msg()
        msg.session_id = session_id
        msg.user_turn_id = int(user_turn_id)
        msg.status = int(status)
        msg.reason = reason
        msg.text = text.strip()
        self.asr_result_pub.publish(msg)

    def destroy_node(self) -> bool:
        if getattr(self, "input_capability", None) is not None and self.input_capability.streaming:
            self._cancel_open_streams("destroy_node")
        self._jobs.put(None)
        self._worker.join(timeout=2.0)
        trace_file = getattr(self, "_trace_file", None)
        if trace_file is not None:
            trace_file.close()
        return super().destroy_node()


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FaAsrNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
