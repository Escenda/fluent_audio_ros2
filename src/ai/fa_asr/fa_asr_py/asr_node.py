#!/usr/bin/env python3
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AsrResult, AudioFrame, TurnContext, VadState
from fa_asr_py.backends.base import AsrBackend, AsrRequest
from fa_asr_py.backends.factory import AsrBackendSettings, build_asr_backend


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

        self.declare_parameter("audio_topic", "audio/frame")
        self.declare_parameter("vad_topic", "voice/vad_state")
        self.declare_parameter("turn_context_topic", "conversation/turn_context")
        self.declare_parameter("asr_result_topic", "voice/asr/result")
        self.declare_parameter("expected_source_id", "")
        self.declare_parameter("expected_stream_id", "")
        self.declare_parameter("target_sample_rate", 16000)
        self.declare_parameter("min_audio_sec", 0.3)
        self.declare_parameter("silence_timeout_sec", 10.0)
        self.declare_parameter("finalize_on_vad_end", True)
        self.declare_parameter("finalize_on_context_inactive", True)
        self.declare_parameter("workspace_dir", "/tmp/fa_asr")
        self.declare_parameter("cleanup_audio_files", True)
        self.declare_parameter("backend.name", "")
        self.declare_parameter("backend.model", "")
        self.declare_parameter("backend.command", "")
        self.declare_parameter("backend.model_path", "")
        self.declare_parameter("backend.openai_realtime.api_key_env", "")
        self.declare_parameter("backend.openai_transcriptions.api_key_env", "")
        self.declare_parameter("backend.language", "ja")
        self.declare_parameter("backend.timeout_sec", 120.0)
        self.declare_parameter("backend.working_directory", "")
        self.declare_parameter("backend.args", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("backend.health_args", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("backend.output_text_path", "")

        self.audio_topic = self._string_parameter("audio_topic")
        self.vad_topic = self._string_parameter("vad_topic")
        self.turn_context_topic = self._string_parameter("turn_context_topic")
        self.asr_result_topic = self._string_parameter("asr_result_topic")
        self.expected_source_id = self._string_parameter("expected_source_id").strip()
        if not self.expected_source_id:
            raise RuntimeError("expected_source_id is required")
        self.expected_stream_id = self._string_parameter("expected_stream_id").strip()
        if not self.expected_stream_id:
            raise RuntimeError("expected_stream_id is required")
        self.target_sample_rate = self._integer_parameter("target_sample_rate")
        self.min_audio_sec = self._double_parameter("min_audio_sec")
        self.silence_timeout_sec = self._double_parameter("silence_timeout_sec")
        self.finalize_on_vad_end = self._bool_parameter("finalize_on_vad_end")
        self.finalize_on_context_inactive = self._bool_parameter(
            "finalize_on_context_inactive"
        )
        self.workspace_dir = Path(self._string_parameter("workspace_dir")).expanduser()
        self.cleanup_audio_files = self._bool_parameter("cleanup_audio_files")

        self.backend = self._load_backend()

        qos_audio = QoSProfile(depth=20)
        qos_audio.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_audio.history = HistoryPolicy.KEEP_LAST

        qos_control = QoSProfile(depth=50)
        qos_control.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_control.history = HistoryPolicy.KEEP_LAST

        qos_reliable = QoSProfile(depth=10)
        qos_reliable.reliability = ReliabilityPolicy.RELIABLE
        qos_reliable.history = HistoryPolicy.KEEP_LAST

        self.asr_result_pub = self.create_publisher(
            AsrResult, self.asr_result_topic, qos_reliable
        )
        self.audio_sub = self.create_subscription(
            AudioFrame, self.audio_topic, self.on_audio, qos_audio
        )
        self.vad_sub = self.create_subscription(
            VadState, self.vad_topic, self.on_vad, qos_control
        )
        self.turn_context_sub = self.create_subscription(
            TurnContext, self.turn_context_topic, self.on_turn_context, qos_reliable
        )
        self.timer = self.create_timer(0.5, self._check_timeout)

        self._active_session_id = ""
        self._active_user_turn_id = 0
        self._context_active = False
        self._last_speech = self.get_clock().now()
        self._samples: list[float] = []
        self._samples_lock = threading.Lock()
        self._jobs: queue.Queue[TranscriptionJob | None] = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            "fa_asr started: audio=%s expected_source_id=%s expected_stream_id=%s vad=%s turn_context=%s result=%s target_sr=%d backend.name=%s",
            self.audio_topic,
            self.expected_source_id,
            self.expected_stream_id,
            self.vad_topic,
            self.turn_context_topic,
            self.asr_result_topic,
            self.target_sample_rate,
            self.backend.name,
        )

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
            )
        )

    def _backend_args(self) -> tuple[str, ...]:
        return self._string_array_parameter("backend.args")

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
        return tuple(parameter.get_parameter_value().string_array_value)

    def on_turn_context(self, msg: TurnContext) -> None:
        if not msg.active or not msg.session_id:
            if self._context_active and self.finalize_on_context_inactive:
                self._submit_current_buffer("context_inactive", publish_timeout_if_empty=False)
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
                "ASR turn started: session=%s user_turn_id=%d",
                new_session_id,
                new_user_turn_id,
            )

        self._active_session_id = new_session_id
        self._active_user_turn_id = new_user_turn_id
        self._context_active = True

    def on_vad(self, msg: VadState) -> None:
        if not self._context_active:
            return
        try:
            self._validate_vad_identity(
                msg,
                expected_source_id=self.expected_source_id,
                expected_stream_id=self.expected_stream_id,
            )
        except ValueError as exc:
            self.get_logger().error("Dropping invalid VadState: %s", exc)
            return
        if msg.is_speech:
            self._last_speech = self.get_clock().now()
        if self.finalize_on_vad_end and msg.end:
            self._submit_current_buffer("vad_end", publish_timeout_if_empty=False)

    def on_audio(self, msg: AudioFrame) -> None:
        if not self._context_active or not self._active_session_id:
            return
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
            self.get_logger().error("Dropping invalid AudioFrame: %s", exc)
            return
        with self._samples_lock:
            self._samples.extend(samples.tolist())

    def _check_timeout(self) -> None:
        if not self._context_active or self.silence_timeout_sec <= 0.0:
            return
        elapsed = (self.get_clock().now() - self._last_speech).nanoseconds / 1e9
        if elapsed < self.silence_timeout_sec:
            return
        self.get_logger().info("ASR silence timeout: %.1fs", elapsed)
        self._submit_current_buffer("silence_timeout", publish_timeout_if_empty=True)
        self._last_speech = self.get_clock().now()

    def _submit_current_buffer(self, reason: str, *, publish_timeout_if_empty: bool) -> None:
        if not self._active_session_id:
            return
        with self._samples_lock:
            samples = np.asarray(self._samples, dtype=np.float32)
            self._samples.clear()

        duration_sec = float(samples.size) / float(self.target_sample_rate)
        if duration_sec < self.min_audio_sec:
            if publish_timeout_if_empty:
                self._publish_result(
                    self._active_session_id,
                    self._active_user_turn_id,
                    AsrResult.STATUS_TIMEOUT,
                    reason,
                    "",
                )
            else:
                self.get_logger().debug(
                    "ASR buffer too short for %s: %.3fs", reason, duration_sec
                )
            return

        self._jobs.put(
            TranscriptionJob(
                session_id=self._active_session_id,
                user_turn_id=self._active_user_turn_id,
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
            transcript = self.backend.transcribe(
                AsrRequest(
                    session_id=job.session_id,
                    user_turn_id=job.user_turn_id,
                    samples=job.samples,
                    sample_rate=job.sample_rate,
                )
            )
            self._publish_result(
                job.session_id,
                job.user_turn_id,
                AsrResult.STATUS_FINAL,
                job.reason,
                transcript,
            )
        except TimeoutError:
            self._publish_result(
                job.session_id,
                job.user_turn_id,
                AsrResult.STATUS_ERROR,
                "backend_timeout",
                "",
            )
        except Exception as exc:
            self.get_logger().error("ASR transcription failed: %s", exc)
            self._publish_result(
                job.session_id,
                job.user_turn_id,
                AsrResult.STATUS_ERROR,
                "backend_error",
                "",
            )

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
