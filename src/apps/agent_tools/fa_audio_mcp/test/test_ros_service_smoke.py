import asyncio
import json
import threading
import uuid
from typing import TypeAlias

import rclpy
from rclpy.context import Context
from mcp.server.fastmcp import FastMCP
from rclpy.executors import SingleThreadedExecutor

from fa_interfaces.msg import (
    AudioClipRef,
    AudioModelRef,
    AudioWindowRef,
    ResolvedTimeRange,
    TranscriptSegment,
)
from fa_interfaces.srv import ArchiveAudioWindow, ExportAudioWindow, TranscribeAudio
from fa_audio_mcp.config import ServerConfig
from fa_audio_mcp.json_types import JsonValue
from fa_audio_mcp.scopes import AudioScopeConfig, AudioScopeResolver
from fa_audio_mcp.server import RosAudioTimelineClient, build_mcp_server

JsonObject: TypeAlias = dict[str, JsonValue]


def test_mcp_tools_call_real_ros_services_and_return_timeline_results() -> None:
    context = _RosSmokeContext()
    try:
        context.start()
        mcp = build_mcp_server(
            context.config,
            RosAudioTimelineClient(context.client_node, context.config),
            AudioScopeResolver(context.config.export_scope_config),
            AudioScopeResolver(context.config.archive_scope_config),
            AudioScopeResolver(context.config.transcribe_scope_config),
        )

        transcribe = _call_tool_json(
            mcp,
            "transcribe_audio",
            {"time_range": "10000000000..20000000000"},
        )
        archive = _call_tool_json(
            mcp,
            "archive_audio_window",
            {
                "time_range": "10000000000..20000000000",
                "reason": "operator stop evidence",
                "related_artifact_ids": ["action_12", "video_observation_9"],
            },
        )
        export = _call_tool_json(
            mcp,
            "export_audio_window",
            {"time_range": "10000000000..20000000000"},
        )

        assert context.transcribe_requests == [
            ("10000000000..20000000000", "audio/high_pass/mic")
        ]
        assert transcribe["segments"] == [
            {
                "start_unix_ns": 10_000_000_000,
                "end_unix_ns": 20_000_000_000,
                "text": "stop the robot",
                "speaker_label": "operator",
            }
        ]
        assert transcribe["audio_window_ref"] == {
            "window_id": "asr-window",
            "window_epoch": 3,
            "source_id": "mic",
            "stream_id": "audio/high_pass/mic",
            "time_range": _expected_time_range(),
        }
        assert transcribe["model_ref"] == {
            "backend_name": "smoke-asr",
            "backend_kind": "asr",
            "model_id": "smoke-model",
            "model_path": "/models/smoke.bin",
            "model_version": "1",
            "model_revision": "test",
        }
        assert transcribe["requested_time_range"] == {
            "start_unix_ns": 10_000_000_000,
            "end_unix_ns": 20_000_000_000,
            "spec": "10000000000..20000000000",
        }

        assert context.archive_requests == [
            (
                "10000000000..20000000000",
                "mic",
                "operator stop evidence",
                ("action_12", "video_observation_9"),
            )
        ]
        assert archive["audio_clip_ref"] == _expected_clip_ref()
        assert archive["time_range"] == _expected_time_range()
        assert archive["requested_time_range"] == {
            "start_unix_ns": 10_000_000_000,
            "end_unix_ns": 20_000_000_000,
            "spec": "10000000000..20000000000",
        }

        assert context.export_requests == [
            (
                "10000000000..20000000000",
                "mic",
                "pcm_s16le",
                "wav",
                "audio/wav",
            )
        ]
        assert export["audio_clip_ref"] == _expected_clip_ref()
        assert export["time_range"] == _expected_time_range()
        assert export["requested_time_range"] == {
            "start_unix_ns": 10_000_000_000,
            "end_unix_ns": 20_000_000_000,
            "spec": "10000000000..20000000000",
        }
    finally:
        context.stop()


def _call_tool_json(mcp: FastMCP, name: str, arguments: JsonObject) -> JsonObject:
    result = asyncio.run(mcp.call_tool(name, arguments))
    assert len(result) == 1
    parsed = json.loads(result[0].text)
    assert isinstance(parsed, dict)
    return parsed


def _expected_time_range() -> JsonObject:
    return {
        "start_unix_ns": 10_000_000_000,
        "end_unix_ns": 20_000_000_000,
        "clock": "media",
        "uncertainty_ns": 0,
        "uncertainty_reason": "",
    }


def _expected_clip_ref() -> JsonObject:
    return {
        "clip_id": "clip-smoke",
        "uri": "file:///tmp/clip-smoke.wav",
        "codec": "pcm_s16le",
        "container": "wav",
        "payload_format": "audio/wav",
        "sample_rate": 16000,
        "channels": 1,
        "duration_ns": 10_000_000_000,
        "time_range": _expected_time_range(),
    }


class _RosSmokeContext:
    def __init__(self) -> None:
        self._suffix = "s_" + uuid.uuid4().hex
        self.config = ServerConfig(
            transport="stdio",
            host="127.0.0.1",
            port=9110,
            export_service_name=f"/fa_audio_mcp_smoke/{self._suffix}/export",
            archive_service_name=f"/fa_audio_mcp_smoke/{self._suffix}/archive",
            transcribe_service_name=f"/fa_audio_mcp_smoke/{self._suffix}/transcribe",
            service_timeout_sec=3.0,
            export_scope_config=AudioScopeConfig(mic="mic", default_scope_key="mic"),
            archive_scope_config=AudioScopeConfig(mic="mic", default_scope_key="mic"),
            transcribe_scope_config=AudioScopeConfig(
                mic="audio/high_pass/mic",
                default_scope_key="mic",
            ),
        )
        self.client_node = None
        self.server_node = None
        self._executor = None
        self._thread = None
        self._context = Context()
        self.export_requests: list[tuple[str, str, str, str, str]] = []
        self.archive_requests: list[tuple[str, str, str, tuple[str, ...]]] = []
        self.transcribe_requests: list[tuple[str, str]] = []

    def start(self) -> None:
        rclpy.init(context=self._context)
        self.client_node = rclpy.create_node(
            f"fa_audio_mcp_smoke_client_{self._suffix}",
            context=self._context,
        )
        self.server_node = rclpy.create_node(
            f"fa_audio_mcp_smoke_server_{self._suffix}",
            context=self._context,
        )
        self.server_node.create_service(
            ExportAudioWindow,
            self.config.export_service_name,
            self._handle_export,
        )
        self.server_node.create_service(
            ArchiveAudioWindow,
            self.config.archive_service_name,
            self._handle_archive,
        )
        self.server_node.create_service(
            TranscribeAudio,
            self.config.transcribe_service_name,
            self._handle_transcribe,
        )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self.server_node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.client_node is not None:
            self.client_node.destroy_node()
        if self.server_node is not None:
            self.server_node.destroy_node()
        if rclpy.ok(context=self._context):
            rclpy.shutdown(context=self._context)

    def _handle_export(self, request, response):
        self.export_requests.append(
            (
                request.time_range_spec,
                request.audio_scope,
                request.codec,
                request.container,
                request.payload_format,
            )
        )
        response.success = True
        response.error_code = ExportAudioWindow.Response.ERROR_NONE
        response.message = ""
        response.time_range = _time_range_msg()
        response.audio_clip_ref = _clip_ref_msg()
        return response

    def _handle_archive(self, request, response):
        self.archive_requests.append(
            (
                request.time_range_spec,
                request.audio_scope,
                request.reason,
                tuple(request.related_artifact_ids),
            )
        )
        response.success = True
        response.error_code = ArchiveAudioWindow.Response.ERROR_NONE
        response.message = ""
        response.time_range = _time_range_msg()
        response.audio_clip_ref = _clip_ref_msg()
        return response

    def _handle_transcribe(self, request, response):
        self.transcribe_requests.append((request.time_range_spec, request.audio_scope))
        response.success = True
        response.error_code = TranscribeAudio.Response.ERROR_NONE
        response.message = ""
        response.time_range = _time_range_msg()
        response.audio_window_ref = _audio_window_ref_msg()
        response.model_ref = _audio_model_ref_msg()
        response.segments = [_transcript_segment_msg()]
        return response


def _time_range_msg() -> ResolvedTimeRange:
    time_range = ResolvedTimeRange()
    time_range.start_unix_ns = 10_000_000_000
    time_range.end_unix_ns = 20_000_000_000
    time_range.clock = ResolvedTimeRange.CLOCK_MEDIA
    time_range.uncertainty_ns = 0
    time_range.uncertainty_reason = ""
    return time_range


def _clip_ref_msg() -> AudioClipRef:
    clip_ref = AudioClipRef()
    clip_ref.clip_id = "clip-smoke"
    clip_ref.uri = "file:///tmp/clip-smoke.wav"
    clip_ref.codec = "pcm_s16le"
    clip_ref.container = "wav"
    clip_ref.payload_format = "audio/wav"
    clip_ref.sample_rate = 16000
    clip_ref.channels = 1
    clip_ref.duration_ns = 10_000_000_000
    clip_ref.time_range = _time_range_msg()
    return clip_ref


def _audio_window_ref_msg() -> AudioWindowRef:
    window_ref = AudioWindowRef()
    window_ref.window_id = "asr-window"
    window_ref.window_epoch = 3
    window_ref.source_id = "mic"
    window_ref.stream_id = "audio/high_pass/mic"
    window_ref.time_range = _time_range_msg()
    return window_ref


def _audio_model_ref_msg() -> AudioModelRef:
    model_ref = AudioModelRef()
    model_ref.backend_name = "smoke-asr"
    model_ref.backend_kind = "asr"
    model_ref.model_id = "smoke-model"
    model_ref.model_path = "/models/smoke.bin"
    model_ref.model_version = "1"
    model_ref.model_revision = "test"
    return model_ref


def _transcript_segment_msg() -> TranscriptSegment:
    segment = TranscriptSegment()
    segment.start_unix_ns = 10_000_000_000
    segment.end_unix_ns = 20_000_000_000
    segment.text = "stop the robot"
    segment.speaker_label = "operator"
    return segment
