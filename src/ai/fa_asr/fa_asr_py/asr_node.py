#!/usr/bin/env python3
from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    AsrResult,
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
    AsrRequest,
    AsrTranscript,
    asr_transcript_text,
    build_asr_transcript,
)
from fa_asr_py.backends.factory import AsrBackendSettings, build_asr_backend
from fa_asr_py.timeline import (
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
    samples: np.ndarray
    sample_rate: int
    reason: str


class FaAsrNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_asr")

        self._declare_required_parameters()

        self.audio_topic = self._string_parameter("audio_topic").strip()
        self.vad_topic = self._string_parameter("vad_topic").strip()
        self.turn_context_topic = self._string_parameter("turn_context_topic").strip()
        self.asr_result_topic = self._string_parameter("asr_result_topic").strip()
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
        self.timeline_clock = self._timeline_clock_parameter("timeline.clock")
        self.timeline_window_id = self._string_parameter("timeline.window_id").strip()
        if not self.timeline_window_id:
            raise RuntimeError("timeline.window_id is required")
        self.timeline_window_epoch = self._non_negative_integer_parameter(
            "timeline.window_epoch"
        )
        self.silence_timeout_sec = self._positive_double_parameter("silence_timeout_sec")
        self.finalize_on_vad_end = self._bool_parameter("finalize_on_vad_end")
        self.finalize_on_context_inactive = self._bool_parameter(
            "finalize_on_context_inactive"
        )
        self.workspace_dir = Path(self._string_parameter("workspace_dir")).expanduser()
        self.cleanup_audio_files = self._bool_parameter("cleanup_audio_files")
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
        self._backend_lock = threading.Lock()
        self._timeline = RollingAsrTimeline(
            sample_rate=self.target_sample_rate,
            retention_sec=self.timeline_retention_sec,
        )
        self._timeline_lock = threading.Lock()
        self._turn_state_lock = threading.RLock()
        self._io_callback_group = MutuallyExclusiveCallbackGroup()
        self._transcribe_service_callback_group = MutuallyExclusiveCallbackGroup()

        qos_audio = self._qos_profile(
            depth_parameter="audio.qos.depth",
            reliable_parameter="audio.qos.reliable",
        )
        qos_vad = self._qos_profile(
            depth_parameter="vad.qos.depth",
            reliable_parameter="vad.qos.reliable",
        )
        qos_turn_context = self._qos_profile(
            depth_parameter="turn_context.qos.depth",
            reliable_parameter="turn_context.qos.reliable",
        )
        qos_result = self._qos_profile(
            depth_parameter="result.qos.depth",
            reliable_parameter="result.qos.reliable",
        )

        self.asr_result_pub = self.create_publisher(
            AsrResult, self.asr_result_topic, qos_result
        )
        self.audio_sub = self.create_subscription(
            AudioFrame,
            self.audio_topic,
            self.on_audio,
            qos_audio,
            callback_group=self._io_callback_group,
        )
        self.vad_sub = self.create_subscription(
            VadState,
            self.vad_topic,
            self.on_vad,
            qos_vad,
            callback_group=self._io_callback_group,
        )
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
        self._samples: list[float] = []
        self._samples_lock = threading.Lock()
        self._jobs: queue.Queue[TranscriptionJob | None] = queue.Queue()
        self._fail_closed_triggered = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"fa_asr started: audio={self.audio_topic} "
            f"expected_source_id={self.expected_source_id} "
            f"expected_stream_id={self.expected_stream_id} vad={self.vad_topic} "
            f"turn_context={self.turn_context_topic} result={self.asr_result_topic} "
            f"transcribe_service={self.transcribe_service_name} "
            f"target_sr={self.target_sample_rate} backend.name={self.backend.name}"
        )

    def _declare_required_parameters(self) -> None:
        self.declare_parameter("audio_topic", Parameter.Type.STRING)
        self.declare_parameter("vad_topic", Parameter.Type.STRING)
        self.declare_parameter("turn_context_topic", Parameter.Type.STRING)
        self.declare_parameter("asr_result_topic", Parameter.Type.STRING)
        self.declare_parameter("transcribe_service_name", Parameter.Type.STRING)
        self.declare_parameter("expected_source_id", Parameter.Type.STRING)
        self.declare_parameter("expected_stream_id", Parameter.Type.STRING)
        self.declare_parameter("target_sample_rate", Parameter.Type.INTEGER)
        self.declare_parameter("min_audio_sec", Parameter.Type.DOUBLE)
        self.declare_parameter("timeline.retention_sec", Parameter.Type.DOUBLE)
        self.declare_parameter("timeline.clock", Parameter.Type.STRING)
        self.declare_parameter("timeline.window_id", Parameter.Type.STRING)
        self.declare_parameter("timeline.window_epoch", Parameter.Type.INTEGER)
        self.declare_parameter("silence_timeout_sec", Parameter.Type.DOUBLE)
        self.declare_parameter("finalize_on_vad_end", Parameter.Type.BOOL)
        self.declare_parameter("finalize_on_context_inactive", Parameter.Type.BOOL)
        self.declare_parameter("workspace_dir", Parameter.Type.STRING)
        self.declare_parameter("cleanup_audio_files", Parameter.Type.BOOL)
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
        self.declare_parameter("vad.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("vad.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("turn_context.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("turn_context.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("result.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("result.qos.reliable", Parameter.Type.BOOL)

    def _validate_identity_contract(self) -> None:
        topics = (
            ("audio_topic", self.audio_topic),
            ("vad_topic", self.vad_topic),
            ("turn_context_topic", self.turn_context_topic),
            ("asr_result_topic", self.asr_result_topic),
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

    @staticmethod
    def _same_identity_string(left: str, right: str) -> bool:
        return left.lstrip("/") == right.lstrip("/")

    def _load_backend(self) -> AsrBackend:
        return build_asr_backend(
            AsrBackendSettings(
                name=self._string_parameter("backend.name"),
                command=self._string_parameter("backend.command"),
                model=self._string_parameter("backend.model"),
                model_path=self._string_parameter("backend.model_path"),
                openai_realtime_api_key_env=self._string_parameter(
                    "backend.openai_realtime.api_key_env"
                ),
                openai_transcriptions_api_key_env=self._string_parameter(
                    "backend.openai_transcriptions.api_key_env"
                ),
                language=self._string_parameter("backend.language"),
                args=self._backend_args(),
                health_args=self._string_array_parameter("backend.health_args"),
                timeout_sec=self._double_parameter("backend.timeout_sec"),
                working_directory=self._string_parameter("backend.working_directory"),
                output_text_path=self._string_parameter("backend.output_text_path"),
                workspace_dir=self.workspace_dir,
                cleanup_audio_files=self.cleanup_audio_files,
                result_format=self._string_parameter("backend.result_format"),
            )
        )

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

    def _qos_profile(self, *, depth_parameter: str, reliable_parameter: str) -> QoSProfile:
        depth = FaAsrNode._positive_integer_parameter(self, depth_parameter)
        reliable = FaAsrNode._bool_parameter(self, reliable_parameter)
        qos = QoSProfile(depth=depth)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = (
            ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
        )
        return qos

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

    def on_turn_context(self, msg: TurnContext) -> None:
        with self._turn_state_lock:
            if not msg.active or not msg.session_id:
                if self._context_active and self.finalize_on_context_inactive:
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
            if self._context_active and key_changed:
                self._submit_current_buffer("turn_replaced", publish_timeout_if_empty=False)

            if key_changed:
                self._clear_buffer()
                self._last_speech = self.get_clock().now()
                self.get_logger().info(
                    f"ASR turn started: session={new_session_id} "
                    f"user_turn_id={new_user_turn_id}"
                )

            self._active_session_id = new_session_id
            self._active_user_turn_id = new_user_turn_id
            self._context_active = True

    def on_vad(self, msg: VadState) -> None:
        with self._turn_state_lock:
            if not self._context_active:
                return
            try:
                self._validate_vad_identity(
                    msg,
                    expected_source_id=self.expected_source_id,
                    expected_stream_id=self.expected_stream_id,
                )
            except ValueError as exc:
                self.get_logger().error(f"Dropping invalid VadState: {exc}")
                return
            if msg.is_speech:
                self._last_speech = self.get_clock().now()
            if self.finalize_on_vad_end and msg.end:
                self._submit_current_buffer("vad_end", publish_timeout_if_empty=False)

    def on_audio(self, msg: AudioFrame) -> None:
        try:
            if int(msg.sample_rate) != self.target_sample_rate:
                raise ValueError(
                    "AudioFrame sample_rate must match target_sample_rate "
                    f"{self.target_sample_rate}, got {msg.sample_rate}"
                )
            samples = self._frame_to_float(
                msg,
                expected_source_id=self.expected_source_id,
                expected_stream_id=self.expected_stream_id,
            )
        except ValueError as exc:
            self.get_logger().error(f"Dropping invalid AudioFrame: {exc}")
            return

        try:
            start_unix_ns = self._stamp_to_unix_ns(msg.header.stamp)
            with self._timeline_lock:
                self._timeline.append(start_unix_ns=start_unix_ns, samples=samples)
        except TimelineRangeError as exc:
            self.get_logger().error(f"Dropping AudioFrame from ASR timeline: {exc}")
            return

        with self._turn_state_lock:
            if not self._context_active or not self._active_session_id:
                return
            with self._samples_lock:
                self._samples.extend(samples.tolist())

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
            return self._transcribe_error(response, exc.error_code, str(exc))

        duration_sec = float(timeline_slice.samples.size) / float(timeline_slice.sample_rate)
        if duration_sec < self.min_audio_sec:
            return self._transcribe_error(
                response,
                TranscribeAudio.Response.ERROR_TRANSCRIBE_FAILED,
                "requested audio duration is below min_audio_sec",
            )

        try:
            with self._backend_lock:
                transcript = self.backend.transcribe(
                    AsrRequest(
                        session_id="timeline",
                        user_turn_id=0,
                        samples=timeline_slice.samples,
                        sample_rate=timeline_slice.sample_rate,
                    )
                )
            self._fill_transcribe_success(response, timeline_slice, transcript)
        except TimeoutError:
            return self._transcribe_error(
                response,
                TranscribeAudio.Response.ERROR_TRANSCRIBE_FAILED,
                "ASR backend timed out",
            )
        except Exception as exc:
            self.get_logger().error(f"ASR timeline transcription failed: {exc}")
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
            sample_count=int(timeline_slice.samples.size),
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
                return
            session_id = self._active_session_id
            user_turn_id = self._active_user_turn_id
            with self._samples_lock:
                samples = np.asarray(self._samples, dtype=np.float32)
                self._samples.clear()

        duration_sec = float(samples.size) / float(self.target_sample_rate)
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
            return

        self._jobs.put(
            TranscriptionJob(
                session_id=session_id,
                user_turn_id=user_turn_id,
                samples=samples,
                sample_rate=self.target_sample_rate,
                reason=reason,
            )
        )

    def _clear_buffer(self) -> None:
        with self._samples_lock:
            self._samples.clear()

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._run_transcription(job)

    def _run_transcription(self, job: TranscriptionJob) -> None:
        try:
            with self._backend_lock:
                transcript = self.backend.transcribe(
                    AsrRequest(
                        session_id=job.session_id,
                        user_turn_id=job.user_turn_id,
                        samples=job.samples,
                        sample_rate=job.sample_rate,
                    )
                )
            validated_transcript = build_asr_transcript(
                transcript.segments,
                sample_count=int(job.samples.size),
            )
            self._publish_result(
                job.session_id,
                job.user_turn_id,
                AsrResult.STATUS_FINAL,
                job.reason,
                asr_transcript_text(validated_transcript),
            )
        except TimeoutError:
            self._publish_result(
                job.session_id,
                job.user_turn_id,
                AsrResult.STATUS_ERROR,
                "backend_timeout",
                "",
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
            self._fail_closed(f"ASR backend failed: {exc}")

    def _fail_closed(self, reason: str) -> None:
        if getattr(self, "_fail_closed_triggered", False):
            return
        self._fail_closed_triggered = True
        self.get_logger().fatal(f"ASR fail closed: {reason}")
        rclpy.shutdown()

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
        if int(msg.sample_rate) <= 0:
            raise ValueError(f"AudioFrame sample_rate must be positive, got {msg.sample_rate}")
        if len(msg.data) % np.dtype("<f4").itemsize != 0:
            raise ValueError("AudioFrame float32 data length is not byte-aligned")
        samples = np.frombuffer(bytes(msg.data), dtype="<f4")
        if not np.all(np.isfinite(samples)):
            raise ValueError("AudioFrame contains non-finite samples")
        if np.any(samples < -1.0) or np.any(samples > 1.0):
            raise ValueError("AudioFrame samples must be normalized to [-1.0, 1.0]")
        return samples

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

    @staticmethod
    def _validate_vad_identity(
        msg: VadState,
        *,
        expected_source_id: str,
        expected_stream_id: str,
    ) -> None:
        if not msg.source_id or not msg.stream_id:
            raise ValueError("VadState source_id and stream_id are required")
        if not expected_source_id:
            raise ValueError("expected_source_id is required")
        if not expected_stream_id:
            raise ValueError("expected_stream_id is required")
        if msg.source_id != expected_source_id:
            raise ValueError("VadState source_id must match expected_source_id")
        if msg.stream_id != expected_stream_id:
            raise ValueError("VadState stream_id must match expected_stream_id")

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
        self._jobs.put(None)
        self._worker.join(timeout=2.0)
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
