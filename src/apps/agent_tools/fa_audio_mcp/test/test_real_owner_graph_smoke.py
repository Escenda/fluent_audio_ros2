import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import struct
import sys
import time
from typing import TypeAlias
import uuid
import wave

import pytest
import rclpy
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AudioFrame
from fa_interfaces.srv import ArchiveAudioWindow, ExportAudioWindow, TranscribeAudio
from fa_audio_mcp.config import ServerConfig
from fa_audio_mcp.json_types import JsonValue
from fa_audio_mcp.scopes import AudioScopeConfig, AudioScopeResolver
from fa_audio_mcp.server import RosAudioTimelineClient, build_mcp_server

JsonObject: TypeAlias = dict[str, JsonValue]

_AUDIO_START_SEC = 10
_SAMPLE_RATE = 16_000
_SAMPLE_COUNT = 8_000
_AUDIO_START_NS = 10_000_000_000
_AUDIO_END_NS = 10_500_000_000
_AUDIO_RANGE_SPEC = "10000000000..10500000000"
_EXPECTED_DURATION_NS = _AUDIO_END_NS - _AUDIO_START_NS


@dataclass(frozen=True)
class _ProcStat:
    pid: int
    process_group_id: int
    state: str


def test_mcp_tools_call_real_asr_and_audio_window_owner_nodes(tmp_path: Path) -> None:
    suffix = "s_" + uuid.uuid4().hex
    config = _OwnerGraphConfig(tmp_path, suffix)
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node(f"fa_audio_mcp_real_owner_client_{suffix}", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    processes: list[_ManagedProcess] = []
    try:
        processes = _start_owner_nodes(config)
        _wait_for_graph(node, config, processes)
        _wait_for_owner_services(node, config, processes)
        _publish_owner_frames(node, executor, config)

        mcp = _build_mcp(node, config)
        transcribe = _call_tool_json(
            mcp,
            "transcribe_audio",
            {"time_range": _AUDIO_RANGE_SPEC},
        )
        archive = _call_tool_json(
            mcp,
            "archive_audio_window",
            {
                "time_range": _AUDIO_RANGE_SPEC,
                "reason": "real owner graph smoke",
                "related_artifact_ids": ["owner_graph_smoke"],
            },
        )

        assert transcribe["segments"] == [
            {
                "start_unix_ns": _AUDIO_START_NS,
                "end_unix_ns": _AUDIO_END_NS,
                "text": "real owner transcript",
                "speaker_label": "",
            }
        ]
        assert transcribe["time_range"] == _expected_time_range()
        assert transcribe["audio_window_ref"] == {
            "window_id": "real_owner_asr_window",
            "window_epoch": 11,
            "source_id": "test-mic",
            "stream_id": "audio/high_pass/mic",
            "time_range": _expected_time_range(),
        }
        assert transcribe["model_ref"] == {
            "backend_name": "local_command",
            "backend_kind": "asr",
            "model_id": "real-owner-fake-asr",
            "model_path": str(config.model_path),
            "model_version": "test",
            "model_revision": "real-owner-smoke",
        }

        clip_ref = archive["audio_clip_ref"]
        assert clip_ref["codec"] == "pcm_s16le"
        assert clip_ref["container"] == "wav"
        assert clip_ref["payload_format"] == "audio/wav"
        assert clip_ref["sample_rate"] == _SAMPLE_RATE
        assert clip_ref["channels"] == 1
        assert clip_ref["duration_ns"] == _EXPECTED_DURATION_NS
        assert clip_ref["time_range"] == _expected_time_range()
        assert str(clip_ref["uri"]).startswith("file://")
        assert archive["time_range"] == _expected_time_range()
        assert archive["requested_time_range"] == {
            "start_unix_ns": _AUDIO_START_NS,
            "end_unix_ns": _AUDIO_END_NS,
            "spec": _AUDIO_RANGE_SPEC,
        }
        clip_path = _assert_archive_wav_clip(clip_ref)
        _assert_archive_metadata(clip_ref, clip_path)

        transcribe_error = _call_tool_error(
            mcp,
            "transcribe_audio",
            {"time_range": _AUDIO_RANGE_SPEC, "audio_scope": "system"},
        )
        archive_error = _call_tool_error(
            mcp,
            "archive_audio_window",
            {
                "time_range": _AUDIO_RANGE_SPEC,
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
        _stop_processes(processes)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok(context=context):
            rclpy.shutdown(context=context)


class _OwnerGraphConfig:
    def __init__(self, tmp_path: Path, suffix: str) -> None:
        self.tmp_path = tmp_path
        self.suffix = suffix
        self.asr_audio_topic = f"/fa_audio_mcp_owner/{suffix}/asr_audio"
        self.window_audio_topic = f"/fa_audio_mcp_owner/{suffix}/window_audio"
        self.vad_topic = f"/fa_audio_mcp_owner/{suffix}/vad"
        self.turn_context_topic = f"/fa_audio_mcp_owner/{suffix}/turn_context"
        self.asr_result_topic = f"/fa_audio_mcp_owner/{suffix}/asr_result"
        self.transcribe_service = f"/fa_audio_mcp_owner/{suffix}/transcribe"
        self.export_service = f"/fa_audio_mcp_owner/{suffix}/export"
        self.archive_service = f"/fa_audio_mcp_owner/{suffix}/archive"
        self.archive_dir = tmp_path / "archive"
        self.asr_workspace = tmp_path / "asr_workspace"
        self.model_path = tmp_path / "fake_asr_model.txt"
        self.asr_config_path = tmp_path / "fa_asr.yaml"
        self.window_config_path = tmp_path / "fa_audio_window.yaml"
        self.asr_log_path = tmp_path / "fa_asr.log"
        self.window_log_path = tmp_path / "fa_audio_window.log"


class _ManagedProcess:
    def __init__(self, command: list[str], log_path: Path) -> None:
        self.command = command
        self.log_path = log_path
        self._log_file = log_path.open("wb")
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._process_group_id = self._process.pid
        except BaseException:
            self._log_file.close()
            raise

    def ensure_running(self) -> None:
        return_code = self._process.poll()
        if return_code is None:
            return
        raise RuntimeError(
            f"process exited with {return_code} for {' '.join(self.command)}: "
            f"{self.log_path.read_text(encoding='utf-8')}"
        )

    def stop(self) -> None:
        try:
            self._signal_process_group(signal.SIGTERM)
            live_pids = self._wait_for_live_process_group_members(timeout_sec=3.0)
            if live_pids:
                self._signal_process_group(signal.SIGKILL)
                live_pids = self._wait_for_live_process_group_members(timeout_sec=3.0)
                if live_pids:
                    raise RuntimeError(
                        "live process group members remained after SIGKILL for "
                        f"{' '.join(self.command)}: pids={live_pids}"
                    )
        finally:
            if not self._log_file.closed:
                self._log_file.close()

    def _signal_process_group(self, sig: signal.Signals) -> None:
        try:
            os.killpg(self._process_group_id, sig)
        except ProcessLookupError:
            return

    def _wait_for_live_process_group_members(self, timeout_sec: float) -> list[int]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self._reap_direct_process()
            live_pids = self._live_process_group_member_pids()
            if not live_pids:
                return []
            time.sleep(0.05)
        self._reap_direct_process()
        return self._live_process_group_member_pids()

    def _reap_direct_process(self) -> None:
        try:
            self._process.wait(timeout=0.0)
        except subprocess.TimeoutExpired:
            return

    def _live_process_group_member_pids(self) -> list[int]:
        live_pids: list[int] = []
        for proc_entry in Path("/proc").iterdir():
            if not proc_entry.name.isdecimal():
                continue
            proc_stat = self._read_proc_stat(proc_entry)
            if proc_stat is None:
                continue
            if proc_stat.process_group_id != self._process_group_id:
                continue
            if proc_stat.state == "Z":
                continue
            live_pids.append(proc_stat.pid)
        return live_pids

    def _read_proc_stat(self, proc_entry: Path) -> _ProcStat | None:
        try:
            stat_text = (proc_entry / "stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            # /proc can lose entries between iteration and stat read.
            # Vanished PIDs are not live members of the managed PGID.
            return None
        close_paren_index = stat_text.rfind(")")
        if close_paren_index < 0:
            raise RuntimeError(f"could not parse process stat for {proc_entry}")
        stat_fields = stat_text[close_paren_index + 2 :].split()
        if len(stat_fields) < 3:
            raise RuntimeError(f"process stat did not contain PGID for {proc_entry}")
        return _ProcStat(
            pid=int(proc_entry.name),
            process_group_id=int(stat_fields[2]),
            state=stat_fields[0],
        )


def _start_owner_nodes(config: _OwnerGraphConfig) -> list[_ManagedProcess]:
    _write_owner_configs(config)
    processes: list[_ManagedProcess] = []
    try:
        processes.append(
            _ManagedProcess(
                [
                    "ros2",
                    "run",
                    "fa_asr",
                    "fa_asr_node",
                    "--ros-args",
                    "--params-file",
                    str(config.asr_config_path),
                ],
                config.asr_log_path,
            )
        )
        processes[-1].ensure_running()
        processes.append(
            _ManagedProcess(
                [
                    "ros2",
                    "run",
                    "fa_audio_window",
                    "fa_audio_window_node",
                    "--ros-args",
                    "--params-file",
                    str(config.window_config_path),
                ],
                config.window_log_path,
            )
        )
        processes[-1].ensure_running()
    except BaseException:
        _stop_processes(processes)
        raise
    return processes


def _stop_processes(processes: list[_ManagedProcess]) -> None:
    failures: list[str] = []
    for process in reversed(processes):
        try:
            process.stop()
        except Exception as exc:
            failures.append(f"{' '.join(process.command)}: {exc}")
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(
            f"failed to stop {len(failures)} managed process(es):\n{details}"
        )


def _write_owner_configs(config: _OwnerGraphConfig) -> None:
    repo_root = Path(__file__).resolve().parents[5]
    fake_worker = (
        repo_root
        / "src"
        / "apps"
        / "agent_tools"
        / "fa_audio_mcp"
        / "test"
        / "fixtures"
        / "fake_asr_worker.py"
    )
    config.model_path.write_text("real owner transcript\n", encoding="utf-8")
    config.archive_dir.mkdir(parents=True, exist_ok=True)
    config.asr_config_path.write_text(
        "\n".join(
            [
                "/**:",
                "  ros__parameters:",
                f"    audio_topic: {_yaml_quote(config.asr_audio_topic)}",
                f"    vad_topic: {_yaml_quote(config.vad_topic)}",
                f"    turn_context_topic: {_yaml_quote(config.turn_context_topic)}",
                f"    asr_result_topic: {_yaml_quote(config.asr_result_topic)}",
                f"    transcribe_service_name: {_yaml_quote(config.transcribe_service)}",
                '    expected_source_id: "test-mic"',
                '    expected_stream_id: "audio/high_pass/mic"',
                f"    target_sample_rate: {_SAMPLE_RATE}",
                "    min_audio_sec: 0.3",
                "    timeline.retention_sec: 10.0",
                '    timeline.clock: "media"',
                '    timeline.window_id: "real_owner_asr_window"',
                "    timeline.window_epoch: 11",
                "    silence_timeout_sec: 10.0",
                "    finalize_on_vad_end: true",
                "    finalize_on_context_inactive: true",
                f"    workspace_dir: {_yaml_quote(str(config.asr_workspace))}",
                "    cleanup_audio_files: true",
                '    backend.name: "local_command"',
                '    backend.kind: "asr"',
                '    backend.model: "real-owner-fake-asr"',
                f"    backend.command: {_yaml_quote(sys.executable)}",
                f"    backend.model_path: {_yaml_quote(str(config.model_path))}",
                '    backend.model_version: "test"',
                '    backend.model_revision: "real-owner-smoke"',
                '    backend.openai_realtime.api_key_env: ""',
                '    backend.openai_transcriptions.api_key_env: ""',
                '    backend.language: "ja"',
                "    backend.timeout_sec: 3.0",
                '    backend.working_directory: ""',
                "    backend.args:",
                f"      - {_yaml_quote(str(fake_worker))}",
                '      - "--audio"',
                '      - "{audio}"',
                '      - "--model"',
                '      - "{model}"',
                '      - "--language"',
                '      - "{language}"',
                '      - "--sample-rate"',
                '      - "{sample_rate}"',
                '      - "--expected-sample"',
                '      - "0.125"',
                "    backend.health_args:",
                f"      - {_yaml_quote(str(fake_worker))}",
                '      - "--health"',
                '      - "--model"',
                '      - "{model}"',
                '      - "--language"',
                '      - "{language}"',
                '    backend.output_text_path: ""',
                "    audio.qos.depth: 20",
                "    audio.qos.reliable: true",
                "    vad.qos.depth: 20",
                "    vad.qos.reliable: false",
                "    turn_context.qos.depth: 10",
                "    turn_context.qos.reliable: true",
                "    result.qos.depth: 10",
                "    result.qos.reliable: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config.window_config_path.write_text(
        "\n".join(
            [
                "/**:",
                "  ros__parameters:",
                f"    input_topic: {_yaml_quote(config.window_audio_topic)}",
                f"    service_name: {_yaml_quote(config.export_service)}",
                f"    archive_service_name: {_yaml_quote(config.archive_service)}",
                '    input.source_id: "test-mic"',
                '    input.stream_id: "audio/archive_pcm16/mic"',
                '    expected.encoding: "PCM16LE"',
                f"    expected.sample_rate: {_SAMPLE_RATE}",
                "    expected.channels: 1",
                "    expected.bit_depth: 16",
                '    expected.layout: "interleaved"',
                "    window.retention_seconds: 10",
                '    window.id: "real_owner_audio_window"',
                "    window.epoch: 7",
                '    audio.default_scope: "mic"',
                "    audio.supported_scopes:",
                '      - "mic"',
                f"    export.output_directory: {_yaml_quote(str(config.archive_dir))}",
                '    export.codec: "pcm_s16le"',
                '    export.container: "wav"',
                '    export.payload_format: "audio/wav"',
                "    qos.depth: 20",
                "    qos.reliable: true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_mcp(node: Node, config: _OwnerGraphConfig) -> FastMCP:
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


def _wait_for_graph(
    node: Node,
    config: _OwnerGraphConfig,
    processes: list[_ManagedProcess],
) -> None:
    deadline = time.monotonic() + 10.0
    asr_pub = node.create_publisher(AudioFrame, config.asr_audio_topic, _audio_qos())
    window_pub = node.create_publisher(AudioFrame, config.window_audio_topic, _audio_qos())
    try:
        while time.monotonic() < deadline:
            for process in processes:
                process.ensure_running()
            if asr_pub.get_subscription_count() >= 1 and window_pub.get_subscription_count() >= 1:
                return
            time.sleep(0.05)
        raise RuntimeError("real owner graph subscriptions were not discovered before timeout")
    finally:
        node.destroy_publisher(asr_pub)
        node.destroy_publisher(window_pub)


def _wait_for_owner_services(
    node: Node,
    config: _OwnerGraphConfig,
    processes: list[_ManagedProcess],
) -> None:
    clients = [
        node.create_client(TranscribeAudio, config.transcribe_service),
        node.create_client(ExportAudioWindow, config.export_service),
        node.create_client(ArchiveAudioWindow, config.archive_service),
    ]
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            for process in processes:
                process.ensure_running()
            if all(client.wait_for_service(timeout_sec=0.05) for client in clients):
                return
        raise RuntimeError("real owner services were not available before timeout")
    finally:
        for client in clients:
            node.destroy_client(client)


def _publish_owner_frames(
    node: Node,
    executor: SingleThreadedExecutor,
    config: _OwnerGraphConfig,
) -> None:
    asr_pub = node.create_publisher(AudioFrame, config.asr_audio_topic, _audio_qos())
    window_pub = node.create_publisher(AudioFrame, config.window_audio_topic, _audio_qos())
    try:
        _wait_for_publisher_subscriptions(asr_pub, window_pub)
        asr_pub.publish(_asr_audio_frame())
        window_pub.publish(_window_audio_frame())
        end_time = time.monotonic() + 0.3
        while time.monotonic() < end_time:
            executor.spin_once(timeout_sec=0.05)
    finally:
        node.destroy_publisher(asr_pub)
        node.destroy_publisher(window_pub)


def _wait_for_publisher_subscriptions(asr_pub, window_pub) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if asr_pub.get_subscription_count() >= 1 and window_pub.get_subscription_count() >= 1:
            return
        time.sleep(0.05)
    raise RuntimeError("real owner frame publishers did not discover subscribers")


def _asr_audio_frame() -> AudioFrame:
    return _audio_frame(
        source_id="test-mic",
        stream_id="audio/high_pass/mic",
        encoding="FLOAT32LE",
        bit_depth=32,
        data=_float32le_bytes(0.125, _SAMPLE_COUNT),
    )


def _window_audio_frame() -> AudioFrame:
    return _audio_frame(
        source_id="test-mic",
        stream_id="audio/archive_pcm16/mic",
        encoding="PCM16LE",
        bit_depth=16,
        data=_pcm16le_bytes(1000, _SAMPLE_COUNT),
    )


def _audio_frame(
    *,
    source_id: str,
    stream_id: str,
    encoding: str,
    bit_depth: int,
    data: bytes,
) -> AudioFrame:
    frame = AudioFrame()
    frame.header.stamp.sec = _AUDIO_START_SEC
    frame.header.stamp.nanosec = 0
    frame.source_id = source_id
    frame.stream_id = stream_id
    frame.encoding = encoding
    frame.sample_rate = _SAMPLE_RATE
    frame.channels = 1
    frame.bit_depth = bit_depth
    frame.layout = "interleaved"
    frame.data = data
    frame.epoch = 1
    return frame


def _float32le_bytes(value: float, count: int) -> bytes:
    return struct.pack("<f", value) * count


def _pcm16le_bytes(value: int, count: int) -> bytes:
    return struct.pack("<h", value) * count


def _audio_qos() -> QoSProfile:
    qos = QoSProfile(depth=20)
    qos.reliability = ReliabilityPolicy.RELIABLE
    return qos


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


def _expected_time_range() -> JsonObject:
    return {
        "start_unix_ns": _AUDIO_START_NS,
        "end_unix_ns": _AUDIO_END_NS,
        "clock": "media",
        "uncertainty_ns": 0,
        "uncertainty_reason": "",
    }


def _assert_archive_wav_clip(clip_ref: JsonObject) -> Path:
    clip_path = _audio_clip_path(clip_ref)
    assert clip_path.is_file()
    with wave.open(str(clip_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == _SAMPLE_RATE
        assert wav_file.getsampwidth() == 2
        assert wav_file.getcomptype() == "NONE"
        assert wav_file.getnframes() == _SAMPLE_COUNT
        duration_ns = wav_file.getnframes() * 1_000_000_000 // wav_file.getframerate()
        assert duration_ns == _EXPECTED_DURATION_NS
        frames = wav_file.readframes(_SAMPLE_COUNT + 1)
    assert len(frames) == _SAMPLE_COUNT * 2
    assert frames[:2] == struct.pack("<h", 1000)
    return clip_path


def _assert_archive_metadata(clip_ref: JsonObject, clip_path: Path) -> None:
    metadata_path = Path(str(clip_path) + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["reason"] == "real owner graph smoke"
    assert metadata["related_artifact_ids"] == ["owner_graph_smoke"]
    assert metadata["source_id"] == "test-mic"
    assert metadata["stream_id"] == "audio/archive_pcm16/mic"
    assert metadata["window_id"] == "real_owner_audio_window"
    assert metadata["window_epoch"] == 7
    assert metadata["audio_scope"] == "mic"
    assert metadata["audio_clip_ref"]["uri"] == clip_ref["uri"]
    assert metadata["time_range"] == _expected_time_range()


def _audio_clip_path(clip_ref: JsonObject) -> Path:
    uri = clip_ref["uri"]
    if not isinstance(uri, str):
        raise RuntimeError("audio_clip_ref.uri must be a string")
    if not uri.startswith("file://"):
        raise RuntimeError("audio_clip_ref.uri must be a file URI")
    return Path(uri.removeprefix("file://"))


def _yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
