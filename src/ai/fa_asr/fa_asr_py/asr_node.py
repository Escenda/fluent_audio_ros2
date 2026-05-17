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
        self.declare_parameter("backend.language", "ja")
        self.declare_parameter("backend.timeout_sec", 120.0)
        self.declare_parameter("backend.working_directory", "")
        self.declare_parameter("backend.args", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("backend.output_text_path", "")

        self.audio_topic = str(self.get_parameter("audio_topic").value)
        self.vad_topic = str(self.get_parameter("vad_topic").value)
        self.turn_context_topic = str(self.get_parameter("turn_context_topic").value)
        self.asr_result_topic = str(self.get_parameter("asr_result_topic").value)
        self.target_sample_rate = int(self.get_parameter("target_sample_rate").value)
        self.min_audio_sec = float(self.get_parameter("min_audio_sec").value)
        self.silence_timeout_sec = float(self.get_parameter("silence_timeout_sec").value)
        self.finalize_on_vad_end = bool(self.get_parameter("finalize_on_vad_end").value)
        self.finalize_on_context_inactive = bool(
            self.get_parameter("finalize_on_context_inactive").value
        )
        self.workspace_dir = Path(str(self.get_parameter("workspace_dir").value)).expanduser()
        self.cleanup_audio_files = bool(self.get_parameter("cleanup_audio_files").value)

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
            "fa_asr started: audio=%s vad=%s turn_context=%s result=%s target_sr=%d backend=%s",
            self.audio_topic,
            self.vad_topic,
            self.turn_context_topic,
            self.asr_result_topic,
            self.target_sample_rate,
            self.backend.name,
        )

    def _load_backend(self) -> AsrBackend:
        return build_asr_backend(
            AsrBackendSettings(
                name=str(self.get_parameter("backend.name").value),
                command=str(self.get_parameter("backend.command").value),
                model=str(self.get_parameter("backend.model").value),
                model_path=str(self.get_parameter("backend.model_path").value),
                language=str(self.get_parameter("backend.language").value),
                args=self._backend_args(),
                timeout_sec=float(self.get_parameter("backend.timeout_sec").value),
                working_directory=str(self.get_parameter("backend.working_directory").value),
                output_text_path=str(self.get_parameter("backend.output_text_path").value),
                workspace_dir=self.workspace_dir,
                cleanup_audio_files=self.cleanup_audio_files,
            )
        )

    def _backend_args(self) -> tuple[str, ...]:
        args_value = self.get_parameter(
            "backend.args"
        ).get_parameter_value().string_array_value
        return tuple(str(part) for part in args_value)

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
            samples = self._frame_to_float(msg)
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
    def _frame_to_float(msg: AudioFrame) -> np.ndarray:
        if not msg.data:
            raise ValueError("AudioFrame data is required")
        if not msg.source_id or not msg.stream_id:
            raise ValueError("AudioFrame source_id and stream_id are required")
        if msg.layout != "interleaved":
            raise ValueError(f"AudioFrame layout must be interleaved, got {msg.layout}")
        if int(msg.channels) != 1:
            raise ValueError(f"AudioFrame channels must be 1, got {msg.channels}")
        if int(msg.bit_depth) != 32:
            raise ValueError(f"AudioFrame bit_depth must be 32, got {msg.bit_depth}")
        if int(msg.sample_rate) <= 0:
            raise ValueError(f"AudioFrame sample_rate must be positive, got {msg.sample_rate}")
        if len(msg.data) % np.dtype(np.float32).itemsize != 0:
            raise ValueError("AudioFrame float32 data length is not byte-aligned")
        samples = np.frombuffer(bytes(msg.data), dtype=np.float32)
        if not np.all(np.isfinite(samples)):
            raise ValueError("AudioFrame contains non-finite samples")
        if np.any(samples < -1.0) or np.any(samples > 1.0):
            raise ValueError("AudioFrame samples must be normalized to [-1.0, 1.0]")
        return samples

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
