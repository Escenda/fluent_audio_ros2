from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import signal
import shutil
import struct
import subprocess
import sys
import time
from typing import TypeAlias
import uuid
import wave

import pytest
import yaml


YamlScalar: TypeAlias = str | int | float | bool | None
YamlMapping: TypeAlias = dict[str, "YamlValue"]
YamlSequence: TypeAlias = list["YamlValue"]
YamlValue: TypeAlias = YamlScalar | YamlMapping | YamlSequence

_SAMPLE_RATE = 16_000
_SAMPLE_COUNT = 8_000
_AUDIO_START_SEC = 10
_AUDIO_START_NS = 10_000_000_000
_AUDIO_END_NS = 10_500_000_000
_AUDIO_RANGE_SPEC = "10000000000..10500000000"
_EXPECTED_DURATION_NS = _AUDIO_END_NS - _AUDIO_START_NS
_ASR_SAMPLE_VALUE = 0.125
_WINDOW_SAMPLE_VALUE = 1000
_TRANSCRIPT = "combined launch transcript"
_MODEL_ID = "combined-launch-fake-asr"
_MODEL_VERSION = "test"
_MODEL_REVISION = "combined-launch-smoke"
_DISABLED_SITE_BINDING = "disabled"
_SOURCE_ID = _DISABLED_SITE_BINDING
_ASR_STREAM_ID = "audio/high_pass/mic"
_WINDOW_STREAM_ID = "audio/archive_pcm16/mic"
_ARCHIVE_REASON = "combined launch smoke"
_ARCHIVE_ARTIFACT_ID = "fluent_audio_system_launch"
_FAKE_ASR_WORKER = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--audio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--expected-sample", type=float)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise RuntimeError(f"missing model: {model_path}")
    if not args.language:
        raise RuntimeError("language is required")
    if args.health:
        return 0
    if args.audio is None:
        raise RuntimeError("--audio is required")
    if args.sample_rate is None:
        raise RuntimeError("--sample-rate is required")
    if args.expected_sample is None:
        raise RuntimeError("--expected-sample is required")
    if args.sample_rate <= 0:
        raise RuntimeError("sample rate must be positive")

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise RuntimeError(f"missing audio: {audio_path}")
    audio_bytes = audio_path.read_bytes()
    if not audio_bytes:
        raise RuntimeError("empty audio")
    if len(audio_bytes) % 4 != 0:
        raise RuntimeError("expected float32le audio")
    samples = [sample for (sample,) in struct.iter_unpack("<f", audio_bytes)]
    if not samples:
        raise RuntimeError("empty float32le audio")
    if not all(math.isfinite(sample) for sample in samples):
        raise RuntimeError("expected finite float32le audio")
    if not all(abs(sample - args.expected_sample) <= 1.0e-7 for sample in samples):
        raise RuntimeError("unexpected float32le sample values")

    print(model_path.read_text(encoding="utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


@dataclass(frozen=True)
class _SmokeConfig:
    suffix: str
    asr_node_name: str
    audio_window_node_name: str
    mcp_node_name: str
    asr_audio_topic: str
    window_audio_topic: str
    vad_topic: str
    turn_context_topic: str
    asr_result_topic: str
    transcribe_service: str
    export_service: str
    archive_service: str
    archive_dir: Path
    asr_workspace: Path
    model_path: Path
    fake_worker_path: Path
    asr_params_path: Path
    window_params_path: Path
    mcp_params_path: Path
    owner_config_path: Path
    adapter_config_path: Path
    mcp_port: int


def test_fluent_audio_system_composes_owner_and_mcp_adapter_configs(
    tmp_path: Path,
) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for fluent_audio_system E2E launch")

    # ROS imports are deferred so non-ROS developer shells can collect this skip-only test.
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    from fa_interfaces.msg import AudioFrame
    from fa_interfaces.srv import ArchiveAudioWindow, ExportAudioWindow, TranscribeAudio

    config = _build_smoke_config(tmp_path)
    _write_runtime_files(config)
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node(
        f"fluent_audio_system_combined_smoke_client_{config.suffix}",
        context=context,
    )
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    process = _start_launch_process(ros2, config)
    try:
        _wait_for_mcp_node(node, executor, process, config)
        _wait_for_owner_services(
            node,
            executor,
            process,
            config,
            TranscribeAudio,
            ExportAudioWindow,
            ArchiveAudioWindow,
        )
        _publish_owner_frames(
            node,
            executor,
            process,
            config,
            AudioFrame,
            QoSProfile,
            ReliabilityPolicy,
        )

        transcribe = _call_transcribe(
            node,
            executor,
            process,
            config,
            TranscribeAudio,
        )
        export = _call_export(node, executor, process, config, ExportAudioWindow)
        archive = _call_archive(node, executor, process, config, ArchiveAudioWindow)

        _assert_transcribe_response(transcribe, config, TranscribeAudio)
        export_clip = _assert_audio_clip_response(export, ExportAudioWindow)
        archive_clip = _assert_audio_clip_response(archive, ArchiveAudioWindow)
        _assert_wav_clip(export_clip)
        archive_path = _assert_wav_clip(archive_clip)
        _assert_archive_metadata(archive_clip, archive_path, config)
    except Exception as exc:
        stdout = _stop_process(process)
        raise AssertionError(f"{exc}\n\nros2 launch output:\n{stdout}") from exc
    finally:
        if process.poll() is None:
            _stop_process(process)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok(context=context):
            rclpy.shutdown(context=context)


def _build_smoke_config(tmp_path: Path) -> _SmokeConfig:
    suffix = "s_" + uuid.uuid4().hex
    base = f"/fluent_audio_system_e2e/{suffix}"
    return _SmokeConfig(
        suffix=suffix,
        asr_node_name=f"fa_asr_{suffix}",
        audio_window_node_name=f"fa_audio_window_{suffix}",
        mcp_node_name=f"fa_audio_mcp_server_{suffix}",
        asr_audio_topic=f"{base}/asr_audio",
        window_audio_topic=f"{base}/window_audio",
        vad_topic=f"{base}/vad",
        turn_context_topic=f"{base}/turn_context",
        asr_result_topic=f"{base}/asr_result",
        transcribe_service=f"{base}/transcribe",
        export_service=f"{base}/export",
        archive_service=f"{base}/archive",
        archive_dir=tmp_path / "archive",
        asr_workspace=tmp_path / "asr_workspace",
        model_path=tmp_path / "fake_asr_model.txt",
        fake_worker_path=tmp_path / "fake_asr_worker.py",
        asr_params_path=tmp_path / "fa_asr.params.yaml",
        window_params_path=tmp_path / "fa_audio_window.params.yaml",
        mcp_params_path=tmp_path / "fa_audio_mcp.params.yaml",
        owner_config_path=tmp_path / "combined_owner_system.yaml",
        adapter_config_path=tmp_path / "combined_mcp_adapter_system.yaml",
        mcp_port=_port_for_suffix(suffix),
    )


def _write_runtime_files(config: _SmokeConfig) -> None:
    # A non-empty fa_in_source_id is propagated to source-bound owner nodes by launch.
    # The smoke frames use that same explicit disabled binding to test effective launch behavior.
    config.archive_dir.mkdir(parents=True, exist_ok=True)
    config.model_path.write_text(_TRANSCRIPT + "\n", encoding="utf-8")
    config.fake_worker_path.write_text(_FAKE_ASR_WORKER, encoding="utf-8")
    _write_asr_params(config)
    _write_audio_window_params(config)
    _write_mcp_params(config)
    _write_owner_system_config(config)
    _write_adapter_system_config(config)


def _write_asr_params(config: _SmokeConfig) -> None:
    _write_yaml(
        config.asr_params_path,
        {
            config.asr_node_name: {
                "ros__parameters": {
                    "audio_topic": config.asr_audio_topic,
                    "vad_topic": config.vad_topic,
                    "turn_context_topic": config.turn_context_topic,
                    "asr_result_topic": config.asr_result_topic,
                    "transcribe_service_name": config.transcribe_service,
                    "expected_source_id": _SOURCE_ID,
                    "expected_stream_id": _ASR_STREAM_ID,
                    "target_sample_rate": _SAMPLE_RATE,
                    "min_audio_sec": 0.3,
                    "timeline.retention_sec": 10.0,
                    "timeline.clock": "media",
                    "timeline.window_id": "combined_launch_asr_window",
                    "timeline.window_epoch": 11,
                    "silence_timeout_sec": 10.0,
                    "finalize_on_vad_end": True,
                    "finalize_on_context_inactive": True,
                    "workspace_dir": str(config.asr_workspace),
                    "cleanup_audio_files": True,
                    "backend.name": "local_command",
                    "backend.kind": "asr",
                    "backend.model": _MODEL_ID,
                    "backend.command": sys.executable,
                    "backend.model_path": str(config.model_path),
                    "backend.model_version": _MODEL_VERSION,
                    "backend.model_revision": _MODEL_REVISION,
                    "backend.openai_realtime.api_key_env": "",
                    "backend.openai_transcriptions.api_key_env": "",
                    "backend.language": "ja",
                    "backend.timeout_sec": 3.0,
                    "backend.working_directory": "",
                    "backend.args": [
                        str(config.fake_worker_path),
                        "--audio",
                        "{audio}",
                        "--model",
                        "{model}",
                        "--language",
                        "{language}",
                        "--sample-rate",
                        "{sample_rate}",
                        "--expected-sample",
                        str(_ASR_SAMPLE_VALUE),
                    ],
                    "backend.health_args": [
                        str(config.fake_worker_path),
                        "--health",
                        "--model",
                        "{model}",
                        "--language",
                        "{language}",
                    ],
                    "backend.output_text_path": "",
                    "audio.qos.depth": 20,
                    "audio.qos.reliable": True,
                    "vad.qos.depth": 20,
                    "vad.qos.reliable": False,
                    "turn_context.qos.depth": 10,
                    "turn_context.qos.reliable": True,
                    "result.qos.depth": 10,
                    "result.qos.reliable": True,
                }
            }
        },
    )


def _write_audio_window_params(config: _SmokeConfig) -> None:
    _write_yaml(
        config.window_params_path,
        {
            config.audio_window_node_name: {
                "ros__parameters": {
                    "input_topic": config.window_audio_topic,
                    "service_name": config.export_service,
                    "archive_service_name": config.archive_service,
                    "input.source_id": _SOURCE_ID,
                    "input.stream_id": _WINDOW_STREAM_ID,
                    "expected.encoding": "PCM16LE",
                    "expected.sample_rate": _SAMPLE_RATE,
                    "expected.channels": 1,
                    "expected.bit_depth": 16,
                    "expected.layout": "interleaved",
                    "window.retention_seconds": 10,
                    "window.id": "combined_launch_audio_window",
                    "window.epoch": 7,
                    "audio.default_scope": "mic",
                    "audio.supported_scopes": ["mic"],
                    "export.output_directory": str(config.archive_dir),
                    "export.codec": "pcm_s16le",
                    "export.container": "wav",
                    "export.payload_format": "audio/wav",
                    "qos.depth": 20,
                    "qos.reliable": True,
                }
            }
        },
    )


def _write_mcp_params(config: _SmokeConfig) -> None:
    _write_yaml(
        config.mcp_params_path,
        {
            config.mcp_node_name: {
                "ros__parameters": {},
            }
        },
    )


def _system_delays() -> YamlMapping:
    return {
        "default_start_delay": 0.1,
        "inter_group_delay": 0.0,
    }


def _write_owner_system_config(config: _SmokeConfig) -> None:
    _write_yaml(
        config.owner_config_path,
        {
            "system": _system_delays(),
            "groups": [
                {
                    "id": "voice_frontend",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_asr",
                            "enable": True,
                            "package": "fa_asr",
                            "exec": "fa_asr_node",
                            "node_name": config.asr_node_name,
                            "params_file": str(config.asr_params_path),
                        },
                    ],
                },
                {
                    "id": "streaming",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_audio_window",
                            "enable": True,
                            "package": "fa_audio_window",
                            "exec": "fa_audio_window_node",
                            "node_name": config.audio_window_node_name,
                            "params_file": str(config.window_params_path),
                        },
                    ],
                },
            ],
        },
    )


def _write_adapter_system_config(config: _SmokeConfig) -> None:
    _write_yaml(
        config.adapter_config_path,
        {
            "system": _system_delays(),
            "groups": [
                {
                    "id": "apps",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_audio_mcp",
                            "enable": True,
                            "package": "fa_audio_mcp",
                            "exec": "fa_audio_mcp_server",
                            "node_name": config.mcp_node_name,
                            "params_file": str(config.mcp_params_path),
                            "env": {
                                "FLUENT_AUDIO_MCP_TRANSPORT": "streamable-http",
                                "FLUENT_AUDIO_MCP_HOST": "127.0.0.1",
                                "FLUENT_AUDIO_MCP_PORT": str(config.mcp_port),
                                "FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC": "5.0",
                                "FLUENT_AUDIO_EXPORT_AUDIO_WINDOW_SERVICE": config.export_service,
                                "FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE": config.archive_service,
                                "FLUENT_AUDIO_TRANSCRIBE_AUDIO_SERVICE": config.transcribe_service,
                                "FLUENT_AUDIO_EXPORT_SCOPE_MIC": "mic",
                                "FLUENT_AUDIO_EXPORT_DEFAULT_SCOPE": "mic",
                                "FLUENT_AUDIO_ARCHIVE_SCOPE_MIC": "mic",
                                "FLUENT_AUDIO_ARCHIVE_DEFAULT_SCOPE": "mic",
                                "FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIC": _ASR_STREAM_ID,
                                "FLUENT_AUDIO_TRANSCRIBE_DEFAULT_SCOPE": "mic",
                            },
                        },
                    ],
                },
            ],
        },
    )


def _write_yaml(path: Path, value: YamlValue) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _port_for_suffix(suffix: str) -> int:
    return 20_000 + (int(suffix.removeprefix("s_")[:8], 16) % 40_000)


def _start_launch_process(ros2: str, config: _SmokeConfig) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            ros2,
            "launch",
            "fluent_audio_system",
            "run.py",
            f"config:={config.owner_config_path},{config.adapter_config_path}",
            "fa_in_enabled:=false",
            "fa_out_enabled:=false",
            f"fa_in_source_id:={_DISABLED_SITE_BINDING}",
            f"fa_out_sink_id:={_DISABLED_SITE_BINDING}",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )


def _wait_for_mcp_node(
    node,
    executor,
    process: subprocess.Popen[str],
    config: _SmokeConfig,
) -> None:
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        _ensure_launch_running(process)
        executor.spin_once(timeout_sec=0.05)
        if _has_node_name(node, config.mcp_node_name):
            return
    raise RuntimeError(f"MCP adapter node did not appear: {config.mcp_node_name}")


def _has_node_name(node, node_name: str) -> bool:
    return any(name == node_name for name, _namespace in node.get_node_names_and_namespaces())


def _wait_for_owner_services(
    node,
    executor,
    process: subprocess.Popen[str],
    config: _SmokeConfig,
    transcribe_srv,
    export_srv,
    archive_srv,
) -> None:
    clients = [
        node.create_client(transcribe_srv, config.transcribe_service),
        node.create_client(export_srv, config.export_service),
        node.create_client(archive_srv, config.archive_service),
    ]
    try:
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            _ensure_launch_running(process)
            executor.spin_once(timeout_sec=0.05)
            if all(client.wait_for_service(timeout_sec=0.05) for client in clients):
                return
        raise RuntimeError("owner services did not become available")
    finally:
        for client in clients:
            node.destroy_client(client)


def _publish_owner_frames(
    node,
    executor,
    process: subprocess.Popen[str],
    config: _SmokeConfig,
    audio_frame_cls,
    qos_profile_cls,
    reliability_policy_cls,
) -> None:
    asr_pub = node.create_publisher(
        audio_frame_cls,
        config.asr_audio_topic,
        _audio_qos(qos_profile_cls, reliability_policy_cls),
    )
    window_pub = node.create_publisher(
        audio_frame_cls,
        config.window_audio_topic,
        _audio_qos(qos_profile_cls, reliability_policy_cls),
    )
    try:
        _wait_for_publisher_subscriptions(asr_pub, window_pub, executor, process)
        asr_pub.publish(
            _audio_frame(
                audio_frame_cls,
                source_id=_SOURCE_ID,
                stream_id=_ASR_STREAM_ID,
                encoding="FLOAT32LE",
                bit_depth=32,
                data=_float32le_bytes(_ASR_SAMPLE_VALUE, _SAMPLE_COUNT),
            )
        )
        window_pub.publish(
            _audio_frame(
                audio_frame_cls,
                source_id=_SOURCE_ID,
                stream_id=_WINDOW_STREAM_ID,
                encoding="PCM16LE",
                bit_depth=16,
                data=_pcm16le_bytes(_WINDOW_SAMPLE_VALUE, _SAMPLE_COUNT),
            )
        )
        end_time = time.monotonic() + 0.4
        while time.monotonic() < end_time:
            _ensure_launch_running(process)
            executor.spin_once(timeout_sec=0.05)
    finally:
        node.destroy_publisher(asr_pub)
        node.destroy_publisher(window_pub)


def _wait_for_publisher_subscriptions(
    asr_pub,
    window_pub,
    executor,
    process: subprocess.Popen[str],
) -> None:
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        _ensure_launch_running(process)
        executor.spin_once(timeout_sec=0.05)
        if asr_pub.get_subscription_count() >= 1 and window_pub.get_subscription_count() >= 1:
            return
    raise RuntimeError("owner frame publishers did not discover subscribers")


def _audio_qos(qos_profile_cls, reliability_policy_cls):
    qos = qos_profile_cls(depth=20)
    qos.reliability = reliability_policy_cls.RELIABLE
    return qos


def _audio_frame(
    audio_frame_cls,
    *,
    source_id: str,
    stream_id: str,
    encoding: str,
    bit_depth: int,
    data: bytes,
):
    frame = audio_frame_cls()
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


def _call_transcribe(
    node,
    executor,
    process: subprocess.Popen[str],
    config: _SmokeConfig,
    transcribe_srv,
):
    client = node.create_client(transcribe_srv, config.transcribe_service)
    try:
        request = transcribe_srv.Request()
        request.time_range_spec = _AUDIO_RANGE_SPEC
        request.audio_scope = ""
        return _call_service(executor, process, client, request, "TranscribeAudio")
    finally:
        node.destroy_client(client)


def _call_export(node, executor, process: subprocess.Popen[str], config: _SmokeConfig, export_srv):
    client = node.create_client(export_srv, config.export_service)
    try:
        request = export_srv.Request()
        request.time_range_spec = _AUDIO_RANGE_SPEC
        request.audio_scope = "mic"
        request.codec = "pcm_s16le"
        request.container = "wav"
        request.payload_format = "audio/wav"
        return _call_service(executor, process, client, request, "ExportAudioWindow")
    finally:
        node.destroy_client(client)


def _call_archive(
    node,
    executor,
    process: subprocess.Popen[str],
    config: _SmokeConfig,
    archive_srv,
):
    client = node.create_client(archive_srv, config.archive_service)
    try:
        request = archive_srv.Request()
        request.time_range_spec = _AUDIO_RANGE_SPEC
        request.audio_scope = "mic"
        request.reason = _ARCHIVE_REASON
        request.related_artifact_ids = [_ARCHIVE_ARTIFACT_ID]
        request.codec = "pcm_s16le"
        request.container = "wav"
        request.payload_format = "audio/wav"
        return _call_service(executor, process, client, request, "ArchiveAudioWindow")
    finally:
        node.destroy_client(client)


def _call_service(executor, process: subprocess.Popen[str], client, request, label: str):
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f"{label} service was unavailable")
    future = client.call_async(request)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        _ensure_launch_running(process)
        executor.spin_once(timeout_sec=0.05)
        if future.done():
            response = future.result()
            if response is None:
                raise RuntimeError(f"{label} service returned no response")
            return response
    raise RuntimeError(f"{label} service did not respond before timeout")


def _assert_transcribe_response(response, config: _SmokeConfig, transcribe_srv) -> None:
    assert response.success, response.message
    assert response.error_code == transcribe_srv.Response.ERROR_NONE
    assert response.message == ""
    assert len(response.segments) == 1
    segment = response.segments[0]
    assert segment.start_unix_ns == _AUDIO_START_NS
    assert segment.end_unix_ns == _AUDIO_END_NS
    assert segment.text == _TRANSCRIPT
    assert segment.speaker_label == ""
    _assert_time_range(response.time_range)

    window_ref = response.audio_window_ref
    assert window_ref.window_id == "combined_launch_asr_window"
    assert window_ref.window_epoch == 11
    assert window_ref.source_id == _SOURCE_ID
    assert window_ref.stream_id == _ASR_STREAM_ID
    _assert_time_range(window_ref.time_range)

    model_ref = response.model_ref
    assert model_ref.backend_name == "local_command"
    assert model_ref.backend_kind == "asr"
    assert model_ref.model_id == _MODEL_ID
    assert model_ref.model_path == str(config.model_path)
    assert model_ref.model_version == _MODEL_VERSION
    assert model_ref.model_revision == _MODEL_REVISION


def _assert_audio_clip_response(response, service_cls):
    assert response.success, response.message
    assert response.error_code == service_cls.Response.ERROR_NONE
    assert response.message == ""
    _assert_time_range(response.time_range)
    clip_ref = response.audio_clip_ref
    assert clip_ref.codec == "pcm_s16le"
    assert clip_ref.container == "wav"
    assert clip_ref.payload_format == "audio/wav"
    assert clip_ref.sample_rate == _SAMPLE_RATE
    assert clip_ref.channels == 1
    assert clip_ref.duration_ns == _EXPECTED_DURATION_NS
    _assert_time_range(clip_ref.time_range)
    assert clip_ref.uri.startswith("file://")
    assert clip_ref.clip_id
    return clip_ref


def _assert_time_range(time_range) -> None:
    assert time_range.start_unix_ns == _AUDIO_START_NS
    assert time_range.end_unix_ns == _AUDIO_END_NS
    assert time_range.clock == "media"
    assert time_range.uncertainty_ns == 0
    assert time_range.uncertainty_reason == ""


def _assert_wav_clip(clip_ref) -> Path:
    clip_path = _clip_path(clip_ref)
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
    expected_payload = _pcm16le_bytes(_WINDOW_SAMPLE_VALUE, _SAMPLE_COUNT)
    assert frames == expected_payload
    return clip_path


def _assert_archive_metadata(clip_ref, clip_path: Path, config: _SmokeConfig) -> None:
    metadata_path = Path(str(clip_path) + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema"] == "fluent_audio.archive_metadata.v1"
    assert metadata["operation"] == "archive_audio_window"
    assert metadata["reason"] == _ARCHIVE_REASON
    assert metadata["related_artifact_ids"] == [_ARCHIVE_ARTIFACT_ID]
    assert metadata["source_id"] == _SOURCE_ID
    assert metadata["stream_id"] == _WINDOW_STREAM_ID
    assert metadata["window_id"] == "combined_launch_audio_window"
    assert metadata["window_epoch"] == 7
    assert metadata["audio_scope"] == "mic"
    assert metadata["time_range"] == {
        "start_unix_ns": _AUDIO_START_NS,
        "end_unix_ns": _AUDIO_END_NS,
        "clock": "media",
        "uncertainty_ns": 0,
        "uncertainty_reason": "",
    }
    assert metadata["audio_clip_ref"]["clip_id"] == clip_ref.clip_id
    assert metadata["audio_clip_ref"]["uri"] == clip_ref.uri
    assert metadata["audio_clip_ref"]["codec"] == "pcm_s16le"
    assert metadata["audio_clip_ref"]["container"] == "wav"
    assert metadata["audio_clip_ref"]["payload_format"] == "audio/wav"
    assert metadata["audio_clip_ref"]["sample_rate"] == _SAMPLE_RATE
    assert metadata["audio_clip_ref"]["channels"] == 1
    assert metadata["audio_clip_ref"]["duration_ns"] == _EXPECTED_DURATION_NS
    assert clip_path.parent == config.archive_dir


def _clip_path(clip_ref) -> Path:
    if not clip_ref.uri.startswith("file://"):
        raise RuntimeError("audio_clip_ref.uri must be a file URI")
    return Path(clip_ref.uri.removeprefix("file://"))


def _ensure_launch_running(process: subprocess.Popen[str]) -> None:
    return_code = process.poll()
    if return_code is None:
        return
    raise RuntimeError(f"ros2 launch exited before smoke completed: {return_code}")


def _stop_process(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _stderr = process.communicate(timeout=5)
    return stdout
