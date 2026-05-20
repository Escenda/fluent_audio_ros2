from __future__ import annotations

from threading import Lock

from mcp.server.fastmcp import FastMCP
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from fa_interfaces.srv import ArchiveAudioWindow, ExportAudioWindow, TranscribeAudio

from fa_audio_mcp.config import ServerConfig, load_server_config
from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.json_types import JsonValue
from fa_audio_mcp.requests import (
    ArchiveAudioRequestValues,
    ExportAudioRequestValues,
    TranscribeAudioRequestValues,
    build_archive_audio_request_values,
    build_export_audio_request_values,
    build_transcribe_audio_request_values,
)
from fa_audio_mcp.responses import (
    format_archive_audio_result,
    format_export_audio_result,
    format_transcribe_audio_result,
)
from fa_audio_mcp.scopes import AudioScopeResolver


class RosAudioTimelineClient:
    def __init__(self, node: Node, config: ServerConfig) -> None:
        self._node = node
        self._timeout_sec = config.service_timeout_sec
        self._export_client = node.create_client(
            ExportAudioWindow,
            config.export_service_name,
        )
        self._archive_client = node.create_client(
            ArchiveAudioWindow,
            config.archive_service_name,
        )
        self._transcribe_client = node.create_client(
            TranscribeAudio,
            config.transcribe_service_name,
        )
        self._service_call_lock = Lock()

    def now_unix_ns(self) -> int:
        return self._node.get_clock().now().nanoseconds

    def export_audio_window(self, values: ExportAudioRequestValues):
        request = ExportAudioWindow.Request()
        request.time_range_spec = values.time_range_spec
        request.audio_scope = values.audio_scope
        request.codec = values.codec
        request.container = values.container
        request.payload_format = values.payload_format
        return self._call_service(
            self._export_client,
            request,
            "export_audio_window",
        )

    def archive_audio_window(self, values: ArchiveAudioRequestValues):
        request = ArchiveAudioWindow.Request()
        request.time_range_spec = values.time_range_spec
        request.audio_scope = values.audio_scope
        request.reason = values.reason
        request.related_artifact_ids = values.related_artifact_ids
        request.codec = values.codec
        request.container = values.container
        request.payload_format = values.payload_format
        return self._call_service(
            self._archive_client,
            request,
            "archive_audio_window",
        )

    def transcribe_audio(self, values: TranscribeAudioRequestValues):
        request = TranscribeAudio.Request()
        request.time_range_spec = values.time_range_spec
        request.audio_scope = values.audio_scope
        return self._call_service(
            self._transcribe_client,
            request,
            "transcribe_audio",
        )

    def _call_service(self, client, request, tool_name: str):
        with self._service_call_lock:
            if not client.wait_for_service(timeout_sec=self._timeout_sec):
                raise AudioToolError(
                    "service_unavailable",
                    f"{tool_name} ROS service was not available before timeout",
                )
            future = client.call_async(request)
            executor = SingleThreadedExecutor(context=self._node.context)
            try:
                rclpy.spin_until_future_complete(
                    self._node,
                    future,
                    executor=executor,
                    timeout_sec=self._timeout_sec,
                )
            finally:
                executor.shutdown()
            response = future.result()
        if response is None:
            raise AudioToolError(
                "service_timeout",
                f"{tool_name} ROS service did not return before timeout",
            )
        return response


def build_mcp_server(
    config: ServerConfig,
    ros_client: RosAudioTimelineClient,
    export_scope_resolver: AudioScopeResolver,
    archive_scope_resolver: AudioScopeResolver,
    transcribe_scope_resolver: AudioScopeResolver,
) -> FastMCP:
    mcp = FastMCP("fluent-audio")

    @mcp.tool(structured_output=False)
    def export_audio_window(
        time_range: str,
        audio_scope: str | None = None,
        codec: str | None = None,
        container: str | None = None,
        payload_format: str | None = None,
    ) -> dict[str, JsonValue]:
        values = build_export_audio_request_values(
            time_range=time_range,
            audio_scope=audio_scope,
            scope_resolver=export_scope_resolver,
            codec=codec,
            container=container,
            payload_format=payload_format,
            now_unix_ns=ros_client.now_unix_ns(),
            marker_resolver=config.time_marker_resolver,
        )
        response = ros_client.export_audio_window(values)
        return format_export_audio_result(response, values.time_range)

    @mcp.tool(structured_output=False)
    def archive_audio_window(
        time_range: str,
        reason: str,
        related_artifact_ids: list[str],
        audio_scope: str | None = None,
        codec: str | None = None,
        container: str | None = None,
        payload_format: str | None = None,
    ) -> dict[str, JsonValue]:
        values = build_archive_audio_request_values(
            time_range=time_range,
            audio_scope=audio_scope,
            reason=reason,
            related_artifact_ids=related_artifact_ids,
            scope_resolver=archive_scope_resolver,
            codec=codec,
            container=container,
            payload_format=payload_format,
            now_unix_ns=ros_client.now_unix_ns(),
            marker_resolver=config.time_marker_resolver,
        )
        response = ros_client.archive_audio_window(values)
        return format_archive_audio_result(response, values.time_range)

    @mcp.tool(structured_output=False)
    def transcribe_audio(
        time_range: str,
        audio_scope: str | None = None,
    ) -> dict[str, JsonValue]:
        values = build_transcribe_audio_request_values(
            time_range=time_range,
            audio_scope=audio_scope,
            scope_resolver=transcribe_scope_resolver,
            now_unix_ns=ros_client.now_unix_ns(),
            marker_resolver=config.time_marker_resolver,
        )
        response = ros_client.transcribe_audio(values)
        return format_transcribe_audio_result(response, values.time_range)

    return mcp


def main() -> None:
    config = load_server_config()
    rclpy.init()
    node = rclpy.create_node("fa_audio_mcp_server")
    try:
        ros_client = RosAudioTimelineClient(node, config)
        export_scope_resolver = AudioScopeResolver(config.export_scope_config)
        archive_scope_resolver = AudioScopeResolver(config.archive_scope_config)
        transcribe_scope_resolver = AudioScopeResolver(config.transcribe_scope_config)
        mcp = build_mcp_server(
            config,
            ros_client,
            export_scope_resolver,
            archive_scope_resolver,
            transcribe_scope_resolver,
        )
        if config.transport != "stdio":
            mcp.settings.host = config.host
            mcp.settings.port = config.port
        mcp.run(transport=config.transport)
    finally:
        node.destroy_node()
        rclpy.shutdown()
