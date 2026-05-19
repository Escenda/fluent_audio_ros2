from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import threading
import time

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

from support.owner_graph import (
    ARCHIVE_ARTIFACT_ID,
    ARCHIVE_REASON,
    JsonObject,
    ManagedProcess,
    NOW_RELATIVE_SIM_NOW_NS,
    NOW_RELATIVE_TIME_RANGE_SPEC,
    OwnerGraphConfig,
    assert_archive_result,
    assert_archive_result_for_spec,
    assert_export_result_for_spec,
    assert_transcribe_result,
    assert_transcribe_result_for_spec,
    build_owner_graph_config,
    build_now_relative_owner_graph_config,
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


def test_installed_server_streamable_http_resolves_now_relative_transcribe_audio(
    tmp_path: Path,
) -> None:
    config = build_now_relative_owner_graph_config(tmp_path)
    mcp_port = _free_tcp_port()
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node(
        f"fa_audio_mcp_streamable_http_relative_client_{config.suffix}",
        context=context,
    )
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    clock_pub = node.create_publisher(Clock, "/clock", _clock_qos())
    clock_pump: FixedSimClockPublisher | None = None
    processes: list[ManagedProcess] = []
    try:
        processes = start_owner_nodes(config)
        wait_for_graph(node, config, processes)
        wait_for_owner_services(node, config, processes)
        publish_owner_frames(node, executor, config)

        server_process = _start_installed_mcp_server(
            config,
            tmp_path,
            mcp_port,
            use_sim_time=True,
        )
        processes.append(server_process)
        _wait_for_clock_subscription(clock_pub, server_process)
        clock_pump = FixedSimClockPublisher(
            clock_pub,
            server_process,
            NOW_RELATIVE_SIM_NOW_NS,
        )
        clock_pump.start()

        url = f"http://127.0.0.1:{mcp_port}/mcp"
        asyncio.run(_wait_for_streamable_http_ready(url, server_process))
        asyncio.run(_assert_streamable_http_now_relative_tools(url, config))
        clock_pump.ensure_running()
    finally:
        try:
            if clock_pump is not None:
                clock_pump.stop()
            node.destroy_publisher(clock_pub)
        finally:
            cleanup_owner_graph_smoke_resources(processes, executor, node, context)


def _start_installed_mcp_server(
    config: OwnerGraphConfig,
    tmp_path: Path,
    port: int,
    *,
    use_sim_time: bool = False,
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
    command = ["ros2", "run", "fa_audio_mcp", "fa_audio_mcp_server"]
    if use_sim_time:
        command.extend(["--ros-args", "-p", "use_sim_time:=true"])
    return ManagedProcess(
        command,
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
                {"time_range": config.audio_range_spec},
            )
            transcribe = _tool_result_json(transcribe_result)
            assert_transcribe_result(transcribe, config)

            archive_result = await session.call_tool(
                "archive_audio_window",
                {
                    "time_range": config.audio_range_spec,
                    "reason": ARCHIVE_REASON,
                    "related_artifact_ids": [ARCHIVE_ARTIFACT_ID],
                },
            )
            archive = _tool_result_json(archive_result)
            assert_archive_result(archive)

            unsupported_scope = await session.call_tool(
                "transcribe_audio",
                {
                    "time_range": config.audio_range_spec,
                    "audio_scope": "system",
                },
            )
            assert unsupported_scope.isError is True
            error_text = _tool_result_text(unsupported_scope)
            assert "unsupported_audio_scope" in error_text
            assert "audio_scope must be empty or match expected_stream_id" in error_text


async def _assert_streamable_http_now_relative_tools(
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

            numeric_transcribe_result = await session.call_tool(
                "transcribe_audio",
                {"time_range": config.audio_range_spec},
            )
            numeric_transcribe = _tool_result_json(numeric_transcribe_result)
            assert_transcribe_result(numeric_transcribe, config)

            relative_transcribe_result = await session.call_tool(
                "transcribe_audio",
                {"time_range": NOW_RELATIVE_TIME_RANGE_SPEC},
            )
            relative_transcribe = _tool_result_json(relative_transcribe_result)
            assert_transcribe_result_for_spec(
                relative_transcribe,
                config,
                NOW_RELATIVE_TIME_RANGE_SPEC,
            )
            _assert_requested_now_relative_range(relative_transcribe)

            relative_archive_result = await session.call_tool(
                "archive_audio_window",
                {
                    "time_range": NOW_RELATIVE_TIME_RANGE_SPEC,
                    "reason": ARCHIVE_REASON,
                    "related_artifact_ids": [ARCHIVE_ARTIFACT_ID],
                },
            )
            relative_archive = _tool_result_json(relative_archive_result)
            assert_archive_result_for_spec(
                relative_archive,
                config,
                NOW_RELATIVE_TIME_RANGE_SPEC,
            )
            _assert_requested_now_relative_range(relative_archive)

            relative_export_result = await session.call_tool(
                "export_audio_window",
                {"time_range": NOW_RELATIVE_TIME_RANGE_SPEC},
            )
            relative_export = _tool_result_json(relative_export_result)
            assert_export_result_for_spec(
                relative_export,
                config,
                NOW_RELATIVE_TIME_RANGE_SPEC,
            )
            _assert_requested_now_relative_range(relative_export)


def _assert_requested_now_relative_range(result: JsonObject) -> None:
    requested_range = result["requested_time_range"]
    if not isinstance(requested_range, dict):
        raise RuntimeError("requested_time_range must be a JSON mapping")
    assert requested_range["spec"] == NOW_RELATIVE_TIME_RANGE_SPEC
    assert requested_range["end_unix_ns"] == NOW_RELATIVE_SIM_NOW_NS
    assert (
        requested_range["end_unix_ns"] - requested_range["start_unix_ns"]
        == 10_000_000_000
    )


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


def _wait_for_clock_subscription(
    clock_pub: Publisher,
    process: ManagedProcess,
) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        process.ensure_running()
        if clock_pub.get_subscription_count() >= 1:
            return
        time.sleep(0.05)
    process.ensure_running()
    raise RuntimeError("fa_audio_mcp_server did not subscribe to /clock")


def _clock_qos() -> QoSProfile:
    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    return qos


def _sim_clock_message(sim_now_ns: int) -> Clock:
    message = Clock()
    message.clock.sec = sim_now_ns // 1_000_000_000
    message.clock.nanosec = sim_now_ns % 1_000_000_000
    return message


class FixedSimClockPublisher:
    def __init__(
        self,
        publisher: Publisher,
        process: ManagedProcess,
        sim_now_ns: int,
    ) -> None:
        self._publisher = publisher
        self._process = process
        self._message = _sim_clock_message(sim_now_ns)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="fa_audio_mcp_fixed_sim_clock",
        )
        self._failure: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def ensure_running(self) -> None:
        if self._failure is None:
            return
        raise RuntimeError(f"sim clock publisher failed: {self._failure}")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("sim clock publisher thread did not stop")
        self.ensure_running()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._process.ensure_running()
                self._publisher.publish(self._message)
            except BaseException as exc:
                self._failure = exc
                self._stop_event.set()
                return
            time.sleep(0.02)
