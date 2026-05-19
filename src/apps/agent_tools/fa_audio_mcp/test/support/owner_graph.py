from __future__ import annotations

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

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AudioFrame
from fa_interfaces.srv import ArchiveAudioWindow, ExportAudioWindow, TranscribeAudio
from fa_audio_mcp.json_types import JsonValue

JsonObject: TypeAlias = dict[str, JsonValue]

SAMPLE_RATE = 16_000
SAMPLE_COUNT = 8_000
AUDIO_START_NS = 10_000_000_000
AUDIO_END_NS = 10_500_000_000
AUDIO_RANGE_SPEC = f"{AUDIO_START_NS}..{AUDIO_END_NS}"
NOW_RELATIVE_TIME_RANGE_SPEC = "now-10s..now"
NOW_RELATIVE_SIM_NOW_NS = 30_000_000_000
NOW_RELATIVE_START_NS = 20_000_000_000
NOW_RELATIVE_SAMPLE_COUNT = SAMPLE_RATE * 10
ARCHIVE_REASON = "real owner graph smoke"
ARCHIVE_ARTIFACT_ID = "owner_graph_smoke"


@dataclass(frozen=True)
class ProcStat:
    pid: int
    process_group_id: int
    state: str


@dataclass(frozen=True)
class AudioRangeExpectation:
    start_unix_ns: int
    end_unix_ns: int
    sample_count: int
    duration_ns: int
    requested_spec: str


class OwnerGraphConfig:
    def __init__(
        self,
        tmp_path: Path,
        suffix: str,
        *,
        audio_start_ns: int,
        sample_count: int,
    ) -> None:
        self.tmp_path = tmp_path
        self.suffix = suffix
        self.audio_start_ns = audio_start_ns
        self.sample_count = sample_count
        self.audio_duration_ns = _sample_count_to_duration_ns(sample_count)
        self.audio_end_ns = audio_start_ns + self.audio_duration_ns
        self.audio_range_spec = f"{self.audio_start_ns}..{self.audio_end_ns}"
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


class ManagedProcess:
    def __init__(
        self,
        command: list[str],
        log_path: Path,
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = command
        self.log_path = log_path
        self._log_file = log_path.open("wb")
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=env,
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
            f"{self.log_text()}"
        )

    def log_text(self) -> str:
        if not self._log_file.closed:
            self._log_file.flush()
        return self.log_path.read_text(encoding="utf-8")

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

    def _read_proc_stat(self, proc_entry: Path) -> ProcStat | None:
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
        return ProcStat(
            pid=int(proc_entry.name),
            process_group_id=int(stat_fields[2]),
            state=stat_fields[0],
        )


def build_owner_graph_config(tmp_path: Path) -> OwnerGraphConfig:
    return OwnerGraphConfig(
        tmp_path,
        "s_" + uuid.uuid4().hex,
        audio_start_ns=AUDIO_START_NS,
        sample_count=SAMPLE_COUNT,
    )


def build_now_relative_owner_graph_config(tmp_path: Path) -> OwnerGraphConfig:
    return OwnerGraphConfig(
        tmp_path,
        "s_" + uuid.uuid4().hex,
        audio_start_ns=NOW_RELATIVE_START_NS,
        sample_count=NOW_RELATIVE_SAMPLE_COUNT,
    )


def start_owner_nodes(config: OwnerGraphConfig) -> list[ManagedProcess]:
    _write_owner_configs(config)
    processes: list[ManagedProcess] = []
    try:
        processes.append(
            ManagedProcess(
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
            ManagedProcess(
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
        stop_processes(processes)
        raise
    return processes


def stop_processes(processes: list[ManagedProcess]) -> None:
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


def cleanup_owner_graph_smoke_resources(
    processes: list[ManagedProcess],
    executor: SingleThreadedExecutor,
    node: Node,
    context: Context,
) -> None:
    failures: list[str] = []
    try:
        stop_processes(processes)
    except Exception as exc:
        failures.append(f"process teardown: {exc}")

    try:
        executor.shutdown()
    except Exception as exc:
        failures.append(f"executor shutdown: {exc}")

    try:
        node.destroy_node()
    except Exception as exc:
        failures.append(f"node destroy: {exc}")

    try:
        if rclpy.ok(context=context):
            rclpy.shutdown(context=context)
    except Exception as exc:
        failures.append(f"rclpy shutdown: {exc}")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"owner graph smoke cleanup failed:\n{details}")


def wait_for_graph(
    node: Node,
    config: OwnerGraphConfig,
    processes: list[ManagedProcess],
) -> None:
    deadline = time.monotonic() + 10.0
    asr_pub = node.create_publisher(AudioFrame, config.asr_audio_topic, _audio_qos())
    window_pub = node.create_publisher(
        AudioFrame,
        config.window_audio_topic,
        _audio_qos(),
    )
    try:
        while time.monotonic() < deadline:
            for process in processes:
                process.ensure_running()
            if (
                asr_pub.get_subscription_count() >= 1
                and window_pub.get_subscription_count() >= 1
            ):
                return
            time.sleep(0.05)
        raise RuntimeError(
            "real owner graph subscriptions were not discovered before timeout"
        )
    finally:
        node.destroy_publisher(asr_pub)
        node.destroy_publisher(window_pub)


def wait_for_owner_services(
    node: Node,
    config: OwnerGraphConfig,
    processes: list[ManagedProcess],
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


def publish_owner_frames(
    node: Node,
    executor: SingleThreadedExecutor,
    config: OwnerGraphConfig,
) -> None:
    asr_pub = node.create_publisher(AudioFrame, config.asr_audio_topic, _audio_qos())
    window_pub = node.create_publisher(
        AudioFrame,
        config.window_audio_topic,
        _audio_qos(),
    )
    try:
        _wait_for_publisher_subscriptions(asr_pub, window_pub)
        asr_pub.publish(_asr_audio_frame(config))
        window_pub.publish(_window_audio_frame(config))
        end_time = time.monotonic() + 0.3
        while time.monotonic() < end_time:
            executor.spin_once(timeout_sec=0.05)
    finally:
        node.destroy_publisher(asr_pub)
        node.destroy_publisher(window_pub)


def assert_transcribe_result(
    transcribe: JsonObject,
    config: OwnerGraphConfig,
) -> None:
    assert_transcribe_result_for_spec(transcribe, config, config.audio_range_spec)


def assert_transcribe_result_for_spec(
    transcribe: JsonObject,
    config: OwnerGraphConfig,
    requested_spec: str,
) -> None:
    assert transcribe["segments"] == [
        {
            "start_unix_ns": config.audio_start_ns,
            "end_unix_ns": config.audio_end_ns,
            "text": "real owner transcript",
            "speaker_label": "",
        }
    ]
    assert transcribe["time_range"] == expected_time_range(config)
    assert transcribe["requested_time_range"] == expected_requested_time_range(
        config,
        requested_spec,
    )
    assert transcribe["audio_window_ref"] == {
        "window_id": "real_owner_asr_window",
        "window_epoch": 11,
        "source_id": "test-mic",
        "stream_id": "audio/high_pass/mic",
        "time_range": expected_time_range(config),
    }
    assert transcribe["model_ref"] == {
        "backend_name": "local_command",
        "backend_kind": "asr",
        "model_id": "real-owner-fake-asr",
        "model_path": str(config.model_path),
        "model_version": "test",
        "model_revision": "real-owner-smoke",
    }


def assert_archive_result(archive: JsonObject) -> Path:
    expectation = _audio_range_expectation(
        start_unix_ns=AUDIO_START_NS,
        sample_count=SAMPLE_COUNT,
        requested_spec=AUDIO_RANGE_SPEC,
    )
    return _assert_archive_result_for_expectation(archive, expectation)


def assert_archive_result_for_spec(
    archive: JsonObject,
    config: OwnerGraphConfig,
    requested_spec: str,
) -> Path:
    expectation = _audio_range_expectation(
        start_unix_ns=config.audio_start_ns,
        sample_count=config.sample_count,
        requested_spec=requested_spec,
    )
    clip_ref, clip_path = _assert_audio_clip_result_for_expectation(
        archive,
        expectation,
    )
    assert_archive_metadata(clip_ref, clip_path, expectation)
    return clip_path


def assert_export_result_for_spec(
    export: JsonObject,
    config: OwnerGraphConfig,
    requested_spec: str,
) -> Path:
    expectation = _audio_range_expectation(
        start_unix_ns=config.audio_start_ns,
        sample_count=config.sample_count,
        requested_spec=requested_spec,
    )
    _, clip_path = _assert_audio_clip_result_for_expectation(export, expectation)
    return clip_path


def _assert_archive_result_for_expectation(
    archive: JsonObject,
    expectation: AudioRangeExpectation,
) -> Path:
    clip_ref, clip_path = _assert_audio_clip_result_for_expectation(
        archive,
        expectation,
    )
    assert_archive_metadata(clip_ref, clip_path, expectation)
    return clip_path


def _assert_audio_clip_result_for_expectation(
    result: JsonObject,
    expectation: AudioRangeExpectation,
) -> tuple[JsonObject, Path]:
    clip_ref = result["audio_clip_ref"]
    if not isinstance(clip_ref, dict):
        raise RuntimeError("audio_clip_ref must be a JSON mapping")
    assert clip_ref["codec"] == "pcm_s16le"
    assert clip_ref["container"] == "wav"
    assert clip_ref["payload_format"] == "audio/wav"
    assert clip_ref["sample_rate"] == SAMPLE_RATE
    assert clip_ref["channels"] == 1
    assert clip_ref["duration_ns"] == expectation.duration_ns
    assert clip_ref["time_range"] == expected_time_range_for_values(
        expectation.start_unix_ns,
        expectation.end_unix_ns,
    )
    assert str(clip_ref["uri"]).startswith("file://")
    assert result["time_range"] == expected_time_range_for_values(
        expectation.start_unix_ns,
        expectation.end_unix_ns,
    )
    assert result["requested_time_range"] == {
        "start_unix_ns": expectation.start_unix_ns,
        "end_unix_ns": expectation.end_unix_ns,
        "spec": expectation.requested_spec,
    }
    clip_path = assert_archive_wav_clip(
        clip_ref,
        sample_count=expectation.sample_count,
        expected_duration_ns=expectation.duration_ns,
    )
    return clip_ref, clip_path


def assert_archive_wav_clip(
    clip_ref: JsonObject,
    *,
    sample_count: int,
    expected_duration_ns: int,
) -> Path:
    clip_path = _audio_clip_path(clip_ref)
    assert clip_path.is_file()
    with wave.open(str(clip_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getsampwidth() == 2
        assert wav_file.getcomptype() == "NONE"
        assert wav_file.getnframes() == sample_count
        duration_ns = wav_file.getnframes() * 1_000_000_000 // wav_file.getframerate()
        assert duration_ns == expected_duration_ns
        frames = wav_file.readframes(sample_count + 1)
    assert len(frames) == sample_count * 2
    assert frames[:2] == struct.pack("<h", 1000)
    return clip_path


def assert_archive_metadata(
    clip_ref: JsonObject,
    clip_path: Path,
    expectation: AudioRangeExpectation,
) -> None:
    metadata_path = Path(str(clip_path) + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["reason"] == ARCHIVE_REASON
    assert metadata["related_artifact_ids"] == [ARCHIVE_ARTIFACT_ID]
    assert metadata["source_id"] == "test-mic"
    assert metadata["stream_id"] == "audio/archive_pcm16/mic"
    assert metadata["window_id"] == "real_owner_audio_window"
    assert metadata["window_epoch"] == 7
    assert metadata["audio_scope"] == "mic"
    assert metadata["audio_clip_ref"]["uri"] == clip_ref["uri"]
    assert metadata["time_range"] == expected_time_range_for_values(
        expectation.start_unix_ns,
        expectation.end_unix_ns,
    )


def expected_time_range(config: OwnerGraphConfig) -> JsonObject:
    return expected_time_range_for_values(config.audio_start_ns, config.audio_end_ns)


def expected_time_range_for_values(start_unix_ns: int, end_unix_ns: int) -> JsonObject:
    return {
        "start_unix_ns": start_unix_ns,
        "end_unix_ns": end_unix_ns,
        "clock": "media",
        "uncertainty_ns": 0,
        "uncertainty_reason": "",
    }


def expected_requested_time_range(
    config: OwnerGraphConfig,
    requested_spec: str,
) -> JsonObject:
    return {
        "start_unix_ns": config.audio_start_ns,
        "end_unix_ns": config.audio_end_ns,
        "spec": requested_spec,
    }


def _write_owner_configs(config: OwnerGraphConfig) -> None:
    repo_root = Path(__file__).resolve().parents[6]
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
                f"    target_sample_rate: {SAMPLE_RATE}",
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
                f"    expected.sample_rate: {SAMPLE_RATE}",
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


def _wait_for_publisher_subscriptions(
    asr_pub: Publisher,
    window_pub: Publisher,
) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if (
            asr_pub.get_subscription_count() >= 1
            and window_pub.get_subscription_count() >= 1
        ):
            return
        time.sleep(0.05)
    raise RuntimeError("real owner frame publishers did not discover subscribers")


def _asr_audio_frame(config: OwnerGraphConfig) -> AudioFrame:
    return _audio_frame(
        config=config,
        source_id="test-mic",
        stream_id="audio/high_pass/mic",
        encoding="FLOAT32LE",
        bit_depth=32,
        data=_float32le_bytes(0.125, config.sample_count),
    )


def _window_audio_frame(config: OwnerGraphConfig) -> AudioFrame:
    return _audio_frame(
        config=config,
        source_id="test-mic",
        stream_id="audio/archive_pcm16/mic",
        encoding="PCM16LE",
        bit_depth=16,
        data=_pcm16le_bytes(1000, config.sample_count),
    )


def _audio_frame(
    *,
    config: OwnerGraphConfig,
    source_id: str,
    stream_id: str,
    encoding: str,
    bit_depth: int,
    data: bytes,
) -> AudioFrame:
    frame = AudioFrame()
    frame.header.stamp.sec = config.audio_start_ns // 1_000_000_000
    frame.header.stamp.nanosec = config.audio_start_ns % 1_000_000_000
    frame.source_id = source_id
    frame.stream_id = stream_id
    frame.encoding = encoding
    frame.sample_rate = SAMPLE_RATE
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


def _sample_count_to_duration_ns(sample_count: int) -> int:
    numerator = sample_count * 1_000_000_000
    duration_ns, remainder = divmod(numerator, SAMPLE_RATE)
    if remainder != 0:
        raise ValueError("sample_count must map exactly to nanoseconds")
    return duration_ns


def _audio_range_expectation(
    *,
    start_unix_ns: int,
    sample_count: int,
    requested_spec: str,
) -> AudioRangeExpectation:
    duration_ns = _sample_count_to_duration_ns(sample_count)
    return AudioRangeExpectation(
        start_unix_ns=start_unix_ns,
        end_unix_ns=start_unix_ns + duration_ns,
        sample_count=sample_count,
        duration_ns=duration_ns,
        requested_spec=requested_spec,
    )


def _audio_qos() -> QoSProfile:
    qos = QoSProfile(depth=20)
    qos.reliability = ReliabilityPolicy.RELIABLE
    return qos


def _audio_clip_path(clip_ref: JsonObject) -> Path:
    uri = clip_ref["uri"]
    if not isinstance(uri, str):
        raise RuntimeError("audio_clip_ref.uri must be a string")
    if not uri.startswith("file://"):
        raise RuntimeError("audio_clip_ref.uri must be a file URI")
    return Path(uri.removeprefix("file://"))


def _yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
