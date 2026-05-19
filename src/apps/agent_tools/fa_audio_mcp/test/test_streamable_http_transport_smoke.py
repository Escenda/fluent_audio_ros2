from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import time

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor

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


def test_installed_server_streamable_http_transport_calls_real_owner_nodes(
    tmp_path: Path,
) -> None:
    config = build_owner_graph_config(tmp_path)
    mcp_port = _free_tcp_port()
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node(
        f"fa_audio_mcp_streamable_http_client_{config.suffix}",
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

        server_process = _start_installed_mcp_server(config, tmp_path, mcp_port)
        processes.append(server_process)
        url = f"http://127.0.0.1:{mcp_port}/mcp"
        asyncio.run(_wait_for_streamable_http_ready(url, server_process))
        asyncio.run(_assert_streamable_http_tools(url, config))
    finally:
        cleanup_owner_graph_smoke_resources(processes, executor, node, context)


def _start_installed_mcp_server(
    config: OwnerGraphConfig,
    tmp_path: Path,
    port: int,
) -> ManagedProcess:
    env = os.environ.copy()
    env.update(
        {
            "FLUENT_AUDIO_MCP_TRANSPORT": "streamable-http",
            "FLUENT_AUDIO_MCP_HOST": "127.0.0.1",
            "FLUENT_AUDIO_MCP_PORT": str(port),
            "FLUENT_AUDIO_EXPORT_AUDIO_WINDOW_SERVICE": config.export_service,
            "FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE": config.archive_service,
            "FLUENT_AUDIO_TRANSCRIBE_AUDIO_SERVICE": config.transcribe_service,
            "FLUENT_AUDIO_EXPORT_SCOPE_MIC": "mic",
            "FLUENT_AUDIO_EXPORT_DEFAULT_SCOPE": "mic",
            "FLUENT_AUDIO_ARCHIVE_SCOPE_MIC": "mic",
            "FLUENT_AUDIO_ARCHIVE_SCOPE_SYSTEM": "system",
            "FLUENT_AUDIO_ARCHIVE_DEFAULT_SCOPE": "mic",
            "FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIC": "audio/high_pass/mic",
            "FLUENT_AUDIO_TRANSCRIBE_SCOPE_SYSTEM": "audio/system",
            "FLUENT_AUDIO_TRANSCRIBE_DEFAULT_SCOPE": "mic",
            "FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC": "5.0",
        }
    )
    return ManagedProcess(
        ["ros2", "run", "fa_audio_mcp", "fa_audio_mcp_server"],
        tmp_path / "fa_audio_mcp_server.log",
        env=env,
    )


async def _wait_for_streamable_http_ready(
    url: str,
    process: ManagedProcess,
) -> None:
    deadline = time.monotonic() + 10.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        process.ensure_running()
        try:
            async with streamable_http_client(url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.1)
    process.ensure_running()
    raise RuntimeError(
        "streamable-http MCP server did not initialize before timeout: "
        f"{last_error}\n\nserver log:\n{process.log_text()}"
    )


async def _assert_streamable_http_tools(
    url: str,
    config: OwnerGraphConfig,
) -> None:
    async with streamable_http_client(url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            assert "transcribe_audio" in tool_names
            assert "archive_audio_window" in tool_names

            transcribe_result = await session.call_tool(
                "transcribe_audio",
                {"time_range": AUDIO_RANGE_SPEC},
            )
            transcribe = _tool_result_json(transcribe_result)
            assert_transcribe_result(transcribe, config)

            archive_result = await session.call_tool(
                "archive_audio_window",
                {
                    "time_range": AUDIO_RANGE_SPEC,
                    "reason": ARCHIVE_REASON,
                    "related_artifact_ids": [ARCHIVE_ARTIFACT_ID],
                },
            )
            archive = _tool_result_json(archive_result)
            assert_archive_result(archive)

            unsupported_scope = await session.call_tool(
                "transcribe_audio",
                {
                    "time_range": AUDIO_RANGE_SPEC,
                    "audio_scope": "system",
                },
            )
            assert unsupported_scope.isError is True
            error_text = _tool_result_text(unsupported_scope)
            assert "unsupported_audio_scope" in error_text
            assert "audio_scope must be empty or match expected_stream_id" in error_text


def _tool_result_json(result: types.CallToolResult) -> JsonObject:
    assert not result.isError
    text = _tool_result_text(result)
    data = json.loads(text)
    assert isinstance(data, dict)
    return data


def _tool_result_text(result: types.CallToolResult) -> str:
    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, types.TextContent)
    return content.text


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
