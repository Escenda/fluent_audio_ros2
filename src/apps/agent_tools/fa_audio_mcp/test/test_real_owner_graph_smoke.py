from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import rclpy
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from support.owner_graph import (
    ARCHIVE_ARTIFACT_ID,
    ARCHIVE_REASON,
    AUDIO_RANGE_SPEC,
    JsonObject,
    ManagedProcess,
    OwnerGraphConfig,
    assert_archive_result,
    assert_transcribe_result,
    build_owner_graph_config,
    cleanup_owner_graph_smoke_resources,
    publish_owner_frames,
    start_owner_nodes,
    wait_for_graph,
    wait_for_owner_services,
)

from fa_audio_mcp.config import ServerConfig
from fa_audio_mcp.scopes import AudioScopeConfig, AudioScopeResolver
from fa_audio_mcp.server import RosAudioTimelineClient, build_mcp_server


def test_mcp_tools_call_real_asr_and_audio_window_owner_nodes(tmp_path: Path) -> None:
    config = build_owner_graph_config(tmp_path)
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node(
        f"fa_audio_mcp_real_owner_client_{config.suffix}",
        context=context,
    )
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    processes: list[ManagedProcess] = []
    try:
        processes = start_owner_nodes(config)
        wait_for_graph(node, config, processes)
        wait_for_owner_services(node, config, processes)
        publish_owner_frames(node, executor, config)

        mcp = _build_mcp(node, config)
        transcribe = _call_tool_json(
            mcp,
            "transcribe_audio",
            {"time_range": AUDIO_RANGE_SPEC},
        )
        archive = _call_tool_json(
            mcp,
            "archive_audio_window",
            {
                "time_range": AUDIO_RANGE_SPEC,
                "reason": ARCHIVE_REASON,
                "related_artifact_ids": [ARCHIVE_ARTIFACT_ID],
            },
        )

        assert_transcribe_result(transcribe, config)
        assert_archive_result(archive)

        transcribe_error = _call_tool_error(
            mcp,
            "transcribe_audio",
            {"time_range": AUDIO_RANGE_SPEC, "audio_scope": "system"},
        )
        archive_error = _call_tool_error(
            mcp,
            "archive_audio_window",
            {
                "time_range": AUDIO_RANGE_SPEC,
                "audio_scope": "system",
                "reason": "real owner graph failure",
                "related_artifact_ids": [],
            },
        )
        assert "unsupported_audio_scope" in transcribe_error
        assert "audio_scope must be empty or match expected_stream_id" in transcribe_error
        assert "unsupported_audio_scope" in archive_error
        assert "audio_scope is not in configured supported set" in archive_error
    finally:
        cleanup_owner_graph_smoke_resources(processes, executor, node, context)


def _build_mcp(node: Node, config: OwnerGraphConfig) -> FastMCP:
    server_config = ServerConfig(
        transport="stdio",
        host="127.0.0.1",
        port=9110,
        export_service_name=config.export_service,
        archive_service_name=config.archive_service,
        transcribe_service_name=config.transcribe_service,
        service_timeout_sec=5.0,
        export_scope_config=AudioScopeConfig(
            mic="mic",
            system="system",
            default_scope_key="mic",
        ),
        archive_scope_config=AudioScopeConfig(
            mic="mic",
            system="system",
            default_scope_key="mic",
        ),
        transcribe_scope_config=AudioScopeConfig(
            mic="audio/high_pass/mic",
            system="audio/system",
            default_scope_key="mic",
        ),
    )
    return build_mcp_server(
        server_config,
        RosAudioTimelineClient(node, server_config),
        AudioScopeResolver(server_config.export_scope_config),
        AudioScopeResolver(server_config.archive_scope_config),
        AudioScopeResolver(server_config.transcribe_scope_config),
    )


def _call_tool_json(mcp: FastMCP, name: str, arguments: JsonObject) -> JsonObject:
    result = asyncio.run(mcp.call_tool(name, arguments))
    assert len(result) == 1
    parsed = result[0].text
    if not isinstance(parsed, str):
        raise RuntimeError("MCP text result must be a string")
    data = json.loads(parsed)
    assert isinstance(data, dict)
    return data


def _call_tool_error(mcp: FastMCP, name: str, arguments: JsonObject) -> str:
    with pytest.raises(ToolError) as exc_info:
        asyncio.run(mcp.call_tool(name, arguments))
    return str(exc_info.value)
