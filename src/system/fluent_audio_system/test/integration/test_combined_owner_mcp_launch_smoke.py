from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import json
import os
import signal
import shutil
import socket
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
JsonScalar: TypeAlias = str | int | float | bool | None
JsonMapping: TypeAlias = dict[str, "JsonValue"]
JsonSequence: TypeAlias = list["JsonValue"]
JsonValue: TypeAlias = JsonScalar | JsonMapping | JsonSequence

_SAMPLE_RATE = 16_000
_SAMPLE_COUNT = 8_000
_NSEC_PER_SEC = 1_000_000_000
_AUDIO_START_NS = 10_000_000_000
_AUDIO_END_NS = 10_500_000_000
_AUDIO_RANGE_SPEC = "10000000000..10500000000"
_EXPECTED_DURATION_NS = _AUDIO_END_NS - _AUDIO_START_NS
_RELATIVE_TIME_RANGE_SPEC = "now-10s..now"
_RELATIVE_AUDIO_START_NS = 20_000_000_000
_RELATIVE_AUDIO_END_NS = 30_000_000_000
_RELATIVE_SAMPLE_COUNT = _SAMPLE_RATE * 10
_RELATIVE_EXPECTED_DURATION_NS = _RELATIVE_AUDIO_END_NS - _RELATIVE_AUDIO_START_NS
_ASR_SAMPLE_VALUE = 0.125
_WINDOW_SAMPLE_VALUE = 1000
_PROFILE_ARCHIVE_SAMPLE_VALUE = 4096
_TRANSCRIPT = "combined launch transcript"
_PROFILE_TRANSCRIPT = "so101 profile launch transcript"
_MODEL_ID = "combined-launch-fake-asr"
_MODEL_VERSION = "test"
_MODEL_REVISION = "combined-launch-smoke"
_DISABLED_SITE_BINDING = "disabled"
_SOURCE_ID = _DISABLED_SITE_BINDING
_ASR_STREAM_ID = "audio/high_pass/mic"
_WINDOW_STREAM_ID = "audio/archive_pcm16/mic"
_ARCHIVE_REASON = "combined launch smoke"
_ARCHIVE_ARTIFACT_ID = "fluent_audio_system_launch"
_PROFILE_MCP_PORT = 9110
_REAL_ASR_SMOKE_ENV = "FLUENT_AUDIO_REAL_ASR_SMOKE"
_REAL_ASR_WORKER_ENV = "FLUENT_AUDIO_ASR_WORKER"
_REAL_ASR_MODEL_PATH_ENV = "FLUENT_AUDIO_ASR_MODEL_PATH"
_REAL_ASR_AUDIO_F32_PATH_ENV = "FLUENT_AUDIO_REAL_ASR_AUDIO_F32_PATH"
_REAL_ASR_SAMPLE_RATE_ENV = "FLUENT_AUDIO_REAL_ASR_SAMPLE_RATE"
_REAL_ASR_EXPECTED_TEXT_ENV = "FLUENT_AUDIO_REAL_ASR_EXPECTED_TEXT"
_PROFILE_CONFIG_ARG = (
    "${share:fluent_audio_system}/config/profiles/so101_voice_frontend.yaml,"
    "${share:fluent_audio_system}/config/profiles/so101_agent_audio_tools.yaml"
)
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
_PROFILE_FAKE_ASR_WORKER = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct


def _samples(audio_path: Path) -> list[float]:
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
    if any(sample < -1.0 or sample > 1.0 for sample in samples):
        raise RuntimeError("expected normalized float32le audio")
    if not all(abs(sample - 0.125) <= 1.0e-7 for sample in samples):
        raise RuntimeError("unexpected profile audio sample values")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    health_parser = subparsers.add_parser("health")
    health_parser.add_argument("--model", required=True)
    health_parser.add_argument("--language", required=True)
    transcribe_parser = subparsers.add_parser("transcribe")
    transcribe_parser.add_argument("--audio", required=True)
    transcribe_parser.add_argument("--model", required=True)
    transcribe_parser.add_argument("--sample-rate", type=int, required=True)
    transcribe_parser.add_argument("--language", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise RuntimeError(f"missing model: {model_path}")
    if not args.language:
        raise RuntimeError("language is required")
    if args.command == "health":
        return 0
    if args.sample_rate <= 0:
        raise RuntimeError("sample rate must be positive")
    _samples(Path(args.audio))
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


@dataclass(frozen=True)
class _ProfileSmokeConfig:
    suffix: str
    client_node_name: str
    high_pass_topic: str
    transcribe_service: str
    export_service: str
    archive_service: str
    asr_node_name: str
    audio_window_node_name: str
    mcp_node_name: str
    asr_model_path: Path
    asr_worker_path: Path
    vad_model_dir: Path
    vad_worker_path: Path
    kws_worker_path: Path
    kws_encoder_path: Path
    kws_decoder_path: Path
    kws_joiner_path: Path
    kws_tokens_path: Path
    kws_keywords_path: Path
    turn_model_path: Path
    turn_worker_path: Path
    start_unix_ns: int
    sample_count: int
    mcp_port: int

    @property
    def end_unix_ns(self) -> int:
        return self.start_unix_ns + self.sample_count * _NSEC_PER_SEC // _SAMPLE_RATE

    @property
    def time_range_spec(self) -> str:
        return f"{self.start_unix_ns}..{self.end_unix_ns}"


@dataclass(frozen=True)
class _RealAsrSmokeInput:
    worker_command: str
    model_path: Path
    audio_bytes: bytes
    expected_text: str | None


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
    from rosgraph_msgs.msg import Clock

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
            start_unix_ns=_AUDIO_START_NS,
            sample_count=_SAMPLE_COUNT,
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
        _assert_wav_clip(export_clip, sample_count=_SAMPLE_COUNT)
        _assert_no_metadata_ref(export_clip)
        archive_path = _assert_wav_clip(archive_clip, sample_count=_SAMPLE_COUNT)
        _assert_archive_metadata(archive_clip, archive_path, config)

        mcp_url = f"http://127.0.0.1:{config.mcp_port}/mcp"
        asyncio.run(_wait_for_streamable_http_ready(mcp_url, process))
        clock_pub = _create_sim_clock_publisher(node, executor, process, Clock)
        try:
            _publish_sim_clock(executor, process, clock_pub, Clock)
            asyncio.run(_assert_streamable_http_tools(mcp_url, config))
            _publish_owner_frames(
                node,
                executor,
                process,
                config,
                AudioFrame,
                QoSProfile,
                ReliabilityPolicy,
                start_unix_ns=_RELATIVE_AUDIO_START_NS,
                sample_count=_RELATIVE_SAMPLE_COUNT,
            )
            _publish_sim_clock(executor, process, clock_pub, Clock)
            asyncio.run(
                _assert_generated_owner_adapter_launch_relative_time_smoke(
                    mcp_url,
                    config,
                )
            )
        finally:
            node.destroy_publisher(clock_pub)
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


def test_so101_profile_pair_runs_owner_services_and_mcp_tools(
    tmp_path: Path,
) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for fluent_audio_system profile launch")
    if not _is_tcp_port_free(_PROFILE_MCP_PORT):
        pytest.skip(f"streamable HTTP MCP profile port {_PROFILE_MCP_PORT} is already in use")

    config = _build_profile_smoke_config(tmp_path)
    _write_profile_runtime_files(config)
    domain_id = _isolated_ros_domain_id(config.suffix)
    previous_domain_id = os.environ.get("ROS_DOMAIN_ID")
    os.environ["ROS_DOMAIN_ID"] = domain_id
    try:
        # ROS imports are deferred so non-ROS developer shells can collect this skip-only test.
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        from fa_interfaces.msg import AudioFrame
        from fa_interfaces.srv import ArchiveAudioWindow, ExportAudioWindow, TranscribeAudio

        context = Context()
        rclpy.init(context=context)
        node = rclpy.create_node(config.client_node_name, context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        process = _start_profile_launch_process(ros2, config, domain_id=domain_id)
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
            _publish_profile_high_pass_frames(
                node,
                executor,
                process,
                config,
                AudioFrame,
                QoSProfile,
                ReliabilityPolicy,
            )

            transcribe = _call_profile_transcribe(
                node,
                executor,
                process,
                config,
                TranscribeAudio,
            )
            export = _call_profile_export(
                node,
                executor,
                process,
                config,
                ExportAudioWindow,
            )
            archive = _call_profile_archive(
                node,
                executor,
                process,
                config,
                ArchiveAudioWindow,
            )

            _assert_profile_transcribe_response(transcribe, config, TranscribeAudio)
            export_clip = _assert_profile_audio_clip_response(
                export,
                config,
                ExportAudioWindow,
            )
            archive_clip = _assert_profile_audio_clip_response(
                archive,
                config,
                ArchiveAudioWindow,
            )
            _assert_profile_wav_clip(export_clip, config)
            _assert_no_metadata_ref(export_clip)
            archive_path = _assert_profile_wav_clip(archive_clip, config)
            _assert_profile_archive_metadata(archive_clip, archive_path, config)

            mcp_url = f"http://127.0.0.1:{config.mcp_port}/mcp"
            asyncio.run(_wait_for_streamable_http_ready(mcp_url, process))
            asyncio.run(_assert_profile_streamable_http_tools(mcp_url, config))
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
    finally:
        if previous_domain_id is None:
            os.environ.pop("ROS_DOMAIN_ID", None)
        else:
            os.environ["ROS_DOMAIN_ID"] = previous_domain_id


def test_so101_profile_pair_runs_real_asr_worker_and_model_when_enabled(
    tmp_path: Path,
) -> None:
    if os.environ.get(_REAL_ASR_SMOKE_ENV) != "1":
        pytest.skip(f"{_REAL_ASR_SMOKE_ENV}=1 is required for real ASR smoke")

    real_asr = _load_real_asr_smoke_input()
    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise AssertionError("ros2 executable is required when real ASR smoke is enabled")
    if not _is_tcp_port_free(_PROFILE_MCP_PORT):
        raise AssertionError(
            f"streamable HTTP MCP profile port {_PROFILE_MCP_PORT} is already in use"
        )

    config = _build_real_asr_profile_smoke_config(tmp_path, real_asr)
    _write_profile_non_asr_runtime_files(config)
    domain_id = _isolated_ros_domain_id(config.suffix)
    previous_domain_id = os.environ.get("ROS_DOMAIN_ID")
    os.environ["ROS_DOMAIN_ID"] = domain_id
    try:
        # ROS imports are deferred so non-ROS developer shells can collect this opt-in test.
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        from fa_interfaces.msg import AudioFrame
        from fa_interfaces.srv import ArchiveAudioWindow, ExportAudioWindow, TranscribeAudio

        context = Context()
        rclpy.init(context=context)
        node = rclpy.create_node(config.client_node_name, context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        process = _start_profile_launch_process(
            ros2,
            config,
            domain_id=domain_id,
        )
        try:
            _wait_for_mcp_node(node, executor, process, config, timeout_sec=130.0)
            _wait_for_owner_services(
                node,
                executor,
                process,
                config,
                TranscribeAudio,
                ExportAudioWindow,
                ArchiveAudioWindow,
                timeout_sec=130.0,
            )
            _publish_profile_high_pass_frames(
                node,
                executor,
                process,
                config,
                AudioFrame,
                QoSProfile,
                ReliabilityPolicy,
                audio_data=real_asr.audio_bytes,
                timeout_sec=130.0,
            )

            transcribe = _call_profile_transcribe(
                node,
                executor,
                process,
                config,
                TranscribeAudio,
                timeout_sec=130.0,
            )

            _assert_real_asr_profile_transcribe_response(
                transcribe,
                config,
                real_asr,
                TranscribeAudio,
            )
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
    finally:
        if previous_domain_id is None:
            os.environ.pop("ROS_DOMAIN_ID", None)
        else:
            os.environ["ROS_DOMAIN_ID"] = previous_domain_id


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
        mcp_port=_free_tcp_port(),
    )


def _build_profile_smoke_config(tmp_path: Path) -> _ProfileSmokeConfig:
    suffix = "p_" + uuid.uuid4().hex
    source_index = int(suffix[-8:], 16) % 10_000
    start_unix_ns = 40_000_000_000 + source_index * _NSEC_PER_SEC
    return _ProfileSmokeConfig(
        suffix=suffix,
        client_node_name=f"so101_profile_smoke_client_{suffix}",
        high_pass_topic="audio/high_pass/frame",
        transcribe_service="transcribe_audio",
        export_service="export_audio_window",
        archive_service="archive_audio_window",
        asr_node_name="fa_asr",
        audio_window_node_name="fa_audio_window_mic",
        mcp_node_name="fa_audio_mcp_server",
        asr_model_path=tmp_path / "profile_fake_asr_model.txt",
        asr_worker_path=tmp_path / "profile_fake_asr_worker.py",
        vad_model_dir=tmp_path / "fake_silero_model",
        vad_worker_path=tmp_path / "fake_vad_worker.py",
        kws_worker_path=tmp_path / "fake_kws_worker.py",
        kws_encoder_path=tmp_path / "kws_encoder.onnx",
        kws_decoder_path=tmp_path / "kws_decoder.onnx",
        kws_joiner_path=tmp_path / "kws_joiner.onnx",
        kws_tokens_path=tmp_path / "kws_tokens.txt",
        kws_keywords_path=tmp_path / "kws_keywords.txt",
        turn_model_path=tmp_path / "fake_smart_turn.onnx",
        turn_worker_path=tmp_path / "fake_turn_worker.py",
        start_unix_ns=start_unix_ns,
        sample_count=_SAMPLE_COUNT,
        mcp_port=_PROFILE_MCP_PORT,
    )


def _build_real_asr_profile_smoke_config(
    tmp_path: Path,
    real_asr: _RealAsrSmokeInput,
) -> _ProfileSmokeConfig:
    suffix = "rp_" + uuid.uuid4().hex
    source_index = int(suffix[-8:], 16) % 10_000
    start_unix_ns = 50_000_000_000 + source_index * _NSEC_PER_SEC
    return _ProfileSmokeConfig(
        suffix=suffix,
        client_node_name=f"so101_real_asr_profile_smoke_client_{suffix}",
        high_pass_topic="audio/high_pass/frame",
        transcribe_service="transcribe_audio",
        export_service="export_audio_window",
        archive_service="archive_audio_window",
        asr_node_name="fa_asr",
        audio_window_node_name="fa_audio_window_mic",
        mcp_node_name="fa_audio_mcp_server",
        asr_model_path=real_asr.model_path,
        asr_worker_path=Path(real_asr.worker_command),
        vad_model_dir=tmp_path / "fake_silero_model",
        vad_worker_path=tmp_path / "fake_vad_worker.py",
        kws_worker_path=tmp_path / "fake_kws_worker.py",
        kws_encoder_path=tmp_path / "kws_encoder.onnx",
        kws_decoder_path=tmp_path / "kws_decoder.onnx",
        kws_joiner_path=tmp_path / "kws_joiner.onnx",
        kws_tokens_path=tmp_path / "kws_tokens.txt",
        kws_keywords_path=tmp_path / "kws_keywords.txt",
        turn_model_path=tmp_path / "fake_smart_turn.onnx",
        turn_worker_path=tmp_path / "fake_turn_worker.py",
        start_unix_ns=start_unix_ns,
        sample_count=len(real_asr.audio_bytes) // 4,
        mcp_port=_PROFILE_MCP_PORT,
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


def _write_profile_runtime_files(config: _ProfileSmokeConfig) -> None:
    _write_profile_fake_asr_runtime_files(config)
    _write_profile_non_asr_runtime_files(config)


def _write_profile_fake_asr_runtime_files(config: _ProfileSmokeConfig) -> None:
    config.asr_model_path.write_text(_PROFILE_TRANSCRIPT + "\n", encoding="utf-8")
    config.asr_worker_path.write_text(_PROFILE_FAKE_ASR_WORKER, encoding="utf-8")
    config.asr_worker_path.chmod(0o755)


def _write_profile_non_asr_runtime_files(config: _ProfileSmokeConfig) -> None:
    config.vad_model_dir.mkdir(parents=True, exist_ok=True)
    (config.vad_model_dir / "hubconf.py").write_text("# fake silero hub\n", encoding="utf-8")
    (config.vad_model_dir / "probability.txt").write_text("0.75\n", encoding="utf-8")
    ai_root = Path(__file__).parents[4] / "ai"
    _copy_executable_fixture(
        ai_root / "fa_vad" / "test" / "fixtures" / "fake_vad_worker.py",
        config.vad_worker_path,
    )

    for model_file in (
        config.kws_encoder_path,
        config.kws_decoder_path,
        config.kws_joiner_path,
        config.kws_tokens_path,
        config.kws_keywords_path,
    ):
        model_file.write_text(f"{model_file.name}\n", encoding="utf-8")
    _copy_executable_fixture(
        ai_root / "fa_kws" / "test" / "fixtures" / "fake_kws_worker.py",
        config.kws_worker_path,
    )

    config.turn_model_path.write_text("0.25\n" + ("x" * 2048), encoding="utf-8")
    _copy_executable_fixture(
        ai_root / "fa_turn_detector" / "test" / "fixtures" / "fake_turn_worker.py",
        config.turn_worker_path,
    )


def _copy_executable_fixture(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | 0o111)


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
                "ros__parameters": {
                    "use_sim_time": True,
                },
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


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _is_tcp_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _isolated_ros_domain_id(suffix: str) -> str:
    return str(20 + (int(suffix[-4:], 16) % 80))


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


def _start_profile_launch_process(
    ros2: str,
    config: _ProfileSmokeConfig,
    *,
    domain_id: str,
) -> subprocess.Popen[str]:
    env = _profile_launch_environment(config, domain_id=domain_id)
    return _start_profile_launch_process_with_environment(ros2, env)


def _profile_launch_environment(
    config: _ProfileSmokeConfig,
    *,
    domain_id: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ROS_DOMAIN_ID": domain_id,
            "FLUENT_AUDIO_VAD_MODEL_DIR": str(config.vad_model_dir),
            "FLUENT_AUDIO_VAD_PROVIDER": "cpu",
            "FLUENT_AUDIO_VAD_WORKER": str(config.vad_worker_path),
            "FLUENT_AUDIO_KWS_PROVIDER": "cpu",
            "FLUENT_AUDIO_KWS_WORKER": str(config.kws_worker_path),
            "FLUENT_AUDIO_KWS_ENCODER": str(config.kws_encoder_path),
            "FLUENT_AUDIO_KWS_DECODER": str(config.kws_decoder_path),
            "FLUENT_AUDIO_KWS_JOINER": str(config.kws_joiner_path),
            "FLUENT_AUDIO_KWS_TOKENS": str(config.kws_tokens_path),
            "FLUENT_AUDIO_KWS_KEYWORDS": str(config.kws_keywords_path),
            "FLUENT_AUDIO_ASR_MODEL_PATH": str(config.asr_model_path),
            "FLUENT_AUDIO_ASR_WORKER": str(config.asr_worker_path),
            "FLUENT_AUDIO_TURN_DETECTOR_MODEL": str(config.turn_model_path),
            "FLUENT_AUDIO_TURN_DETECTOR_PROVIDER": "CPUExecutionProvider",
            "FLUENT_AUDIO_TURN_DETECTOR_WORKER": str(config.turn_worker_path),
            "FA_KWS_FAKE_MODE": "none",
        }
    )
    return env


def _start_profile_launch_process_with_environment(
    ros2: str,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            ros2,
            "launch",
            "fluent_audio_system",
            "run.py",
            f"config:={_PROFILE_CONFIG_ARG}",
            "fa_in_enabled:=false",
            "fa_out_enabled:=false",
            f"fa_in_source_id:={_DISABLED_SITE_BINDING}",
            f"fa_out_sink_id:={_DISABLED_SITE_BINDING}",
        ],
        env=env,
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
    *,
    timeout_sec: float = 12.0,
) -> None:
    deadline = time.monotonic() + timeout_sec
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
    *,
    timeout_sec: float = 12.0,
) -> None:
    clients = [
        node.create_client(transcribe_srv, config.transcribe_service),
        node.create_client(export_srv, config.export_service),
        node.create_client(archive_srv, config.archive_service),
    ]
    try:
        deadline = time.monotonic() + timeout_sec
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
    *,
    start_unix_ns: int,
    sample_count: int,
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
                start_unix_ns=start_unix_ns,
                data=_float32le_bytes(_ASR_SAMPLE_VALUE, sample_count),
            )
        )
        window_pub.publish(
            _audio_frame(
                audio_frame_cls,
                source_id=_SOURCE_ID,
                stream_id=_WINDOW_STREAM_ID,
                encoding="PCM16LE",
                bit_depth=16,
                start_unix_ns=start_unix_ns,
                data=_pcm16le_bytes(_WINDOW_SAMPLE_VALUE, sample_count),
            )
        )
        end_time = time.monotonic() + 0.4
        while time.monotonic() < end_time:
            _ensure_launch_running(process)
            executor.spin_once(timeout_sec=0.05)
    finally:
        node.destroy_publisher(asr_pub)
        node.destroy_publisher(window_pub)


def _publish_profile_high_pass_frames(
    node,
    executor,
    process: subprocess.Popen[str],
    config: _ProfileSmokeConfig,
    audio_frame_cls,
    qos_profile_cls,
    reliability_policy_cls,
    *,
    audio_data: bytes | None = None,
    timeout_sec: float = 12.0,
) -> None:
    high_pass_pub = node.create_publisher(
        audio_frame_cls,
        config.high_pass_topic,
        _audio_qos(qos_profile_cls, reliability_policy_cls),
    )
    try:
        _wait_for_profile_high_pass_subscriptions(
            high_pass_pub,
            executor,
            process,
            timeout_sec=timeout_sec,
        )
        frame_data = audio_data
        if frame_data is None:
            frame_data = _float32le_bytes(_ASR_SAMPLE_VALUE, config.sample_count)
        high_pass_pub.publish(
            _audio_frame(
                audio_frame_cls,
                source_id=_SOURCE_ID,
                stream_id=_ASR_STREAM_ID,
                encoding="FLOAT32LE",
                bit_depth=32,
                start_unix_ns=config.start_unix_ns,
                data=frame_data,
            )
        )
        end_time = time.monotonic() + 0.8
        while time.monotonic() < end_time:
            _ensure_launch_running(process)
            executor.spin_once(timeout_sec=0.05)
    finally:
        node.destroy_publisher(high_pass_pub)


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


def _wait_for_profile_high_pass_subscriptions(
    high_pass_pub,
    executor,
    process: subprocess.Popen[str],
    *,
    timeout_sec: float = 12.0,
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        _ensure_launch_running(process)
        executor.spin_once(timeout_sec=0.05)
        if high_pass_pub.get_subscription_count() >= 3:
            return
    raise RuntimeError("profile high-pass publisher did not discover expected subscribers")


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
    start_unix_ns: int,
    data: bytes,
):
    frame = audio_frame_cls()
    start_sec, start_nanosec = divmod(start_unix_ns, _NSEC_PER_SEC)
    frame.header.stamp.sec = start_sec
    frame.header.stamp.nanosec = start_nanosec
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


def _create_sim_clock_publisher(node, executor, process: subprocess.Popen[str], clock_msg_cls):
    clock_pub = node.create_publisher(clock_msg_cls, "/clock", 10)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        _ensure_launch_running(process)
        executor.spin_once(timeout_sec=0.05)
        if clock_pub.get_subscription_count() >= 1:
            return clock_pub
    node.destroy_publisher(clock_pub)
    raise RuntimeError("MCP node did not subscribe to /clock with use_sim_time enabled")


def _publish_sim_clock(
    executor,
    process: subprocess.Popen[str],
    clock_pub,
    clock_msg_cls,
) -> None:
    clock_msg = clock_msg_cls()
    clock_sec, clock_nanosec = divmod(_RELATIVE_AUDIO_END_NS, _NSEC_PER_SEC)
    clock_msg.clock.sec = clock_sec
    clock_msg.clock.nanosec = clock_nanosec
    for _index in range(5):
        _ensure_launch_running(process)
        clock_pub.publish(clock_msg)
        executor.spin_once(timeout_sec=0.02)


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


def _call_profile_transcribe(
    node,
    executor,
    process: subprocess.Popen[str],
    config: _ProfileSmokeConfig,
    transcribe_srv,
    *,
    timeout_sec: float = 8.0,
):
    client = node.create_client(transcribe_srv, config.transcribe_service)
    try:
        request = transcribe_srv.Request()
        request.time_range_spec = config.time_range_spec
        request.audio_scope = ""
        return _call_service(
            executor,
            process,
            client,
            request,
            "TranscribeAudio",
            timeout_sec=timeout_sec,
        )
    finally:
        node.destroy_client(client)


def _call_profile_export(
    node,
    executor,
    process: subprocess.Popen[str],
    config: _ProfileSmokeConfig,
    export_srv,
):
    client = node.create_client(export_srv, config.export_service)
    try:
        request = export_srv.Request()
        request.time_range_spec = config.time_range_spec
        request.audio_scope = "mic"
        request.codec = "pcm_s16le"
        request.container = "wav"
        request.payload_format = "audio/wav"
        return _call_service(executor, process, client, request, "ExportAudioWindow")
    finally:
        node.destroy_client(client)


def _call_profile_archive(
    node,
    executor,
    process: subprocess.Popen[str],
    config: _ProfileSmokeConfig,
    archive_srv,
):
    client = node.create_client(archive_srv, config.archive_service)
    try:
        request = archive_srv.Request()
        request.time_range_spec = config.time_range_spec
        request.audio_scope = "mic"
        request.reason = _ARCHIVE_REASON
        request.related_artifact_ids = [_ARCHIVE_ARTIFACT_ID]
        request.codec = "pcm_s16le"
        request.container = "wav"
        request.payload_format = "audio/wav"
        return _call_service(executor, process, client, request, "ArchiveAudioWindow")
    finally:
        node.destroy_client(client)


def _call_service(
    executor,
    process: subprocess.Popen[str],
    client,
    request,
    label: str,
    *,
    timeout_sec: float = 8.0,
):
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f"{label} service was unavailable")
    future = client.call_async(request)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        _ensure_launch_running(process)
        executor.spin_once(timeout_sec=0.05)
        if future.done():
            response = future.result()
            if response is None:
                raise RuntimeError(f"{label} service returned no response")
            return response
    raise RuntimeError(f"{label} service did not respond before timeout")


async def _wait_for_streamable_http_ready(
    url: str,
    process: subprocess.Popen[str],
) -> None:
    # MCP imports stay deferred so non-ROS shells can collect the ROS-gated smoke.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    deadline = time.monotonic() + 10.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        _ensure_launch_running(process)
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
    _ensure_launch_running(process)
    raise RuntimeError(
        "streamable-http MCP server did not initialize before timeout: "
        f"{last_error}"
    )


async def _assert_streamable_http_tools(url: str, config: _SmokeConfig) -> None:
    # MCP imports stay deferred so non-ROS shells can collect the ROS-gated smoke.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            assert {
                "transcribe_audio",
                "archive_audio_window",
                "export_audio_window",
            } <= tool_names

            transcribe_result = await session.call_tool(
                "transcribe_audio",
                {"time_range": _AUDIO_RANGE_SPEC},
            )
            transcribe = _tool_result_json(transcribe_result)
            _assert_transcribe_json(
                transcribe,
                config,
                start_unix_ns=_AUDIO_START_NS,
                end_unix_ns=_AUDIO_END_NS,
                requested_spec=_AUDIO_RANGE_SPEC,
            )

            archive_result = await session.call_tool(
                "archive_audio_window",
                {
                    "time_range": _AUDIO_RANGE_SPEC,
                    "reason": _ARCHIVE_REASON,
                    "related_artifact_ids": [_ARCHIVE_ARTIFACT_ID],
                },
            )
            archive = _tool_result_json(archive_result)
            archive_clip = _assert_audio_clip_json(
                archive,
                start_unix_ns=_AUDIO_START_NS,
                end_unix_ns=_AUDIO_END_NS,
                expected_duration_ns=_EXPECTED_DURATION_NS,
                requested_spec=_AUDIO_RANGE_SPEC,
            )
            archive_path = _assert_wav_clip_json(archive_clip, sample_count=_SAMPLE_COUNT)
            _assert_archive_metadata_json(
                archive_clip,
                archive_path,
                config,
                start_unix_ns=_AUDIO_START_NS,
                end_unix_ns=_AUDIO_END_NS,
                expected_duration_ns=_EXPECTED_DURATION_NS,
            )

            export_result = await session.call_tool(
                "export_audio_window",
                {"time_range": _AUDIO_RANGE_SPEC},
            )
            export = _tool_result_json(export_result)
            export_clip = _assert_audio_clip_json(
                export,
                start_unix_ns=_AUDIO_START_NS,
                end_unix_ns=_AUDIO_END_NS,
                expected_duration_ns=_EXPECTED_DURATION_NS,
                requested_spec=_AUDIO_RANGE_SPEC,
            )
            _assert_wav_clip_json(export_clip, sample_count=_SAMPLE_COUNT)
            _assert_no_metadata_ref_json(export_clip)

            unsupported_scope = await session.call_tool(
                "transcribe_audio",
                {
                    "time_range": _AUDIO_RANGE_SPEC,
                    "audio_scope": "system",
                },
            )
            assert unsupported_scope.isError is True
            error_text = _tool_result_text(unsupported_scope)
            assert "unsupported_audio_scope" in error_text
            assert "audio_scope 'system' is not configured" in error_text


async def _assert_generated_owner_adapter_launch_relative_time_smoke(
    url: str,
    config: _SmokeConfig,
) -> None:
    # MCP imports stay deferred so non-ROS shells can collect the ROS-gated smoke.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            transcribe_result = await session.call_tool(
                "transcribe_audio",
                {"time_range": _RELATIVE_TIME_RANGE_SPEC},
            )
            transcribe = _tool_result_json(transcribe_result)
            _assert_transcribe_json(
                transcribe,
                config,
                start_unix_ns=_RELATIVE_AUDIO_START_NS,
                end_unix_ns=_RELATIVE_AUDIO_END_NS,
                requested_spec=_RELATIVE_TIME_RANGE_SPEC,
            )
            _assert_requested_time_range_duration(transcribe)

            archive_result = await session.call_tool(
                "archive_audio_window",
                {
                    "time_range": _RELATIVE_TIME_RANGE_SPEC,
                    "reason": _ARCHIVE_REASON,
                    "related_artifact_ids": [_ARCHIVE_ARTIFACT_ID],
                },
            )
            archive = _tool_result_json(archive_result)
            archive_clip = _assert_audio_clip_json(
                archive,
                start_unix_ns=_RELATIVE_AUDIO_START_NS,
                end_unix_ns=_RELATIVE_AUDIO_END_NS,
                expected_duration_ns=_RELATIVE_EXPECTED_DURATION_NS,
                requested_spec=_RELATIVE_TIME_RANGE_SPEC,
            )
            _assert_requested_time_range_duration(archive)
            archive_path = _assert_wav_clip_json(
                archive_clip,
                sample_count=_RELATIVE_SAMPLE_COUNT,
            )
            _assert_archive_metadata_json(
                archive_clip,
                archive_path,
                config,
                start_unix_ns=_RELATIVE_AUDIO_START_NS,
                end_unix_ns=_RELATIVE_AUDIO_END_NS,
                expected_duration_ns=_RELATIVE_EXPECTED_DURATION_NS,
            )

            export_result = await session.call_tool(
                "export_audio_window",
                {"time_range": _RELATIVE_TIME_RANGE_SPEC},
            )
            export = _tool_result_json(export_result)
            export_clip = _assert_audio_clip_json(
                export,
                start_unix_ns=_RELATIVE_AUDIO_START_NS,
                end_unix_ns=_RELATIVE_AUDIO_END_NS,
                expected_duration_ns=_RELATIVE_EXPECTED_DURATION_NS,
                requested_spec=_RELATIVE_TIME_RANGE_SPEC,
            )
            _assert_requested_time_range_duration(export)
            _assert_wav_clip_json(export_clip, sample_count=_RELATIVE_SAMPLE_COUNT)
            _assert_no_metadata_ref_json(export_clip)


async def _assert_profile_streamable_http_tools(
    url: str,
    config: _ProfileSmokeConfig,
) -> None:
    # MCP imports stay deferred so non-ROS shells can collect the ROS-gated smoke.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            assert {
                "transcribe_audio",
                "archive_audio_window",
                "export_audio_window",
            } <= tool_names

            transcribe_result = await session.call_tool(
                "transcribe_audio",
                {"time_range": config.time_range_spec},
            )
            transcribe = _tool_result_json(transcribe_result)
            _assert_profile_transcribe_json(transcribe, config)

            archive_result = await session.call_tool(
                "archive_audio_window",
                {
                    "time_range": config.time_range_spec,
                    "reason": _ARCHIVE_REASON,
                    "related_artifact_ids": [_ARCHIVE_ARTIFACT_ID],
                },
            )
            archive = _tool_result_json(archive_result)
            archive_clip = _assert_profile_audio_clip_json(archive, config)
            archive_path = _assert_profile_wav_clip_json(archive_clip, config)
            _assert_profile_archive_metadata_json(archive_clip, archive_path, config)

            export_result = await session.call_tool(
                "export_audio_window",
                {"time_range": config.time_range_spec},
            )
            export = _tool_result_json(export_result)
            export_clip = _assert_profile_audio_clip_json(export, config)
            _assert_profile_wav_clip_json(export_clip, config)
            _assert_no_metadata_ref_json(export_clip)


def _tool_result_json(result) -> JsonMapping:
    assert not result.isError
    text = _tool_result_text(result)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise RuntimeError("MCP result must be a JSON mapping")
    return data


def _tool_result_text(result) -> str:
    from mcp import types

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, types.TextContent)
    return content.text


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


def _assert_profile_transcribe_response(
    response,
    config: _ProfileSmokeConfig,
    transcribe_srv,
) -> None:
    assert response.success, response.message
    assert response.error_code == transcribe_srv.Response.ERROR_NONE
    assert response.message == ""
    assert len(response.segments) == 1
    segment = response.segments[0]
    assert segment.start_unix_ns == config.start_unix_ns
    assert segment.end_unix_ns == config.end_unix_ns
    assert segment.text == _PROFILE_TRANSCRIPT
    assert segment.speaker_label == ""
    _assert_profile_time_range(response.time_range, config)

    window_ref = response.audio_window_ref
    assert window_ref.window_id == "so101_voice_frontend_mic_asr"
    assert window_ref.window_epoch == 1
    assert window_ref.source_id == _SOURCE_ID
    assert window_ref.stream_id == _ASR_STREAM_ID
    _assert_profile_time_range(window_ref.time_range, config)

    model_ref = response.model_ref
    assert model_ref.backend_name == "whisper.cpp"
    assert model_ref.backend_kind == "asr"
    assert model_ref.model_id == ""
    assert model_ref.model_path == str(config.asr_model_path)
    assert model_ref.model_version == ""
    assert model_ref.model_revision == ""


def _assert_real_asr_profile_transcribe_response(
    response,
    config: _ProfileSmokeConfig,
    real_asr: _RealAsrSmokeInput,
    transcribe_srv,
) -> None:
    assert response.success, response.message
    assert response.error_code == transcribe_srv.Response.ERROR_NONE
    assert response.message == ""
    assert len(response.segments) == 1
    segment = response.segments[0]
    assert segment.start_unix_ns == config.start_unix_ns
    assert segment.end_unix_ns == config.end_unix_ns
    transcript = segment.text.strip()
    if real_asr.expected_text:
        assert real_asr.expected_text in transcript
    else:
        assert transcript
    assert segment.speaker_label == ""
    _assert_profile_time_range(response.time_range, config)

    window_ref = response.audio_window_ref
    assert window_ref.window_id == "so101_voice_frontend_mic_asr"
    assert window_ref.window_epoch == 1
    assert window_ref.source_id == _SOURCE_ID
    assert window_ref.stream_id == _ASR_STREAM_ID
    _assert_profile_time_range(window_ref.time_range, config)

    model_ref = response.model_ref
    assert model_ref.backend_name == "whisper.cpp"
    assert model_ref.backend_kind == "asr"
    assert model_ref.model_path == str(real_asr.model_path)
    assert model_ref.model_id == ""
    assert model_ref.model_version == ""
    assert model_ref.model_revision == ""


def _assert_transcribe_json(
    transcribe: JsonMapping,
    config: _SmokeConfig,
    *,
    start_unix_ns: int,
    end_unix_ns: int,
    requested_spec: str,
) -> None:
    assert transcribe["segments"] == [
        {
            "start_unix_ns": start_unix_ns,
            "end_unix_ns": end_unix_ns,
            "text": _TRANSCRIPT,
            "speaker_label": "",
        }
    ]
    assert transcribe["time_range"] == _expected_time_range_json(start_unix_ns, end_unix_ns)
    assert transcribe["requested_time_range"] == _expected_requested_time_range_json(
        start_unix_ns,
        end_unix_ns,
        requested_spec,
    )
    assert transcribe["audio_window_ref"] == {
        "window_id": "combined_launch_asr_window",
        "window_epoch": 11,
        "source_id": _SOURCE_ID,
        "stream_id": _ASR_STREAM_ID,
        "time_range": _expected_time_range_json(start_unix_ns, end_unix_ns),
    }
    assert transcribe["model_ref"] == {
        "backend_name": "local_command",
        "backend_kind": "asr",
        "model_id": _MODEL_ID,
        "model_path": str(config.model_path),
        "model_version": _MODEL_VERSION,
        "model_revision": _MODEL_REVISION,
    }


def _assert_profile_transcribe_json(
    transcribe: JsonMapping,
    config: _ProfileSmokeConfig,
) -> None:
    assert transcribe["segments"] == [
        {
            "start_unix_ns": config.start_unix_ns,
            "end_unix_ns": config.end_unix_ns,
            "text": _PROFILE_TRANSCRIPT,
            "speaker_label": "",
        }
    ]
    expected_time_range = _expected_time_range_json(
        config.start_unix_ns,
        config.end_unix_ns,
    )
    assert transcribe["time_range"] == expected_time_range
    assert transcribe["requested_time_range"] == _expected_requested_time_range_json(
        config.start_unix_ns,
        config.end_unix_ns,
        config.time_range_spec,
    )
    assert transcribe["audio_window_ref"] == {
        "window_id": "so101_voice_frontend_mic_asr",
        "window_epoch": 1,
        "source_id": _SOURCE_ID,
        "stream_id": _ASR_STREAM_ID,
        "time_range": expected_time_range,
    }
    assert transcribe["model_ref"] == {
        "backend_name": "whisper.cpp",
        "backend_kind": "asr",
        "model_id": "",
        "model_path": str(config.asr_model_path),
        "model_version": "",
        "model_revision": "",
    }


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
    assert clip_ref.content_sha256
    return clip_ref


def _assert_profile_audio_clip_response(
    response,
    config: _ProfileSmokeConfig,
    service_cls,
):
    assert response.success, response.message
    assert response.error_code == service_cls.Response.ERROR_NONE
    assert response.message == ""
    _assert_profile_time_range(response.time_range, config)
    clip_ref = response.audio_clip_ref
    assert clip_ref.codec == "pcm_s16le"
    assert clip_ref.container == "wav"
    assert clip_ref.payload_format == "audio/wav"
    assert clip_ref.sample_rate == _SAMPLE_RATE
    assert clip_ref.channels == 1
    assert clip_ref.duration_ns == config.end_unix_ns - config.start_unix_ns
    _assert_profile_time_range(clip_ref.time_range, config)
    assert clip_ref.uri.startswith("file://")
    assert clip_ref.clip_id
    assert clip_ref.content_sha256
    return clip_ref


def _assert_audio_clip_json(
    result: JsonMapping,
    *,
    start_unix_ns: int,
    end_unix_ns: int,
    expected_duration_ns: int,
    requested_spec: str,
) -> JsonMapping:
    clip_ref = result["audio_clip_ref"]
    if not isinstance(clip_ref, dict):
        raise RuntimeError("audio_clip_ref must be a JSON mapping")
    assert clip_ref["codec"] == "pcm_s16le"
    assert clip_ref["container"] == "wav"
    assert clip_ref["payload_format"] == "audio/wav"
    assert clip_ref["sample_rate"] == _SAMPLE_RATE
    assert clip_ref["channels"] == 1
    assert clip_ref["duration_ns"] == expected_duration_ns
    assert clip_ref["time_range"] == _expected_time_range_json(start_unix_ns, end_unix_ns)
    assert result["time_range"] == _expected_time_range_json(start_unix_ns, end_unix_ns)
    assert result["requested_time_range"] == _expected_requested_time_range_json(
        start_unix_ns,
        end_unix_ns,
        requested_spec,
    )
    assert str(clip_ref["uri"]).startswith("file://")
    assert clip_ref["clip_id"]
    assert clip_ref["content_sha256"]
    return clip_ref


def _assert_profile_audio_clip_json(
    result: JsonMapping,
    config: _ProfileSmokeConfig,
) -> JsonMapping:
    clip_ref = result["audio_clip_ref"]
    if not isinstance(clip_ref, dict):
        raise RuntimeError("audio_clip_ref must be a JSON mapping")
    expected_time_range = _expected_time_range_json(
        config.start_unix_ns,
        config.end_unix_ns,
    )
    assert clip_ref["codec"] == "pcm_s16le"
    assert clip_ref["container"] == "wav"
    assert clip_ref["payload_format"] == "audio/wav"
    assert clip_ref["sample_rate"] == _SAMPLE_RATE
    assert clip_ref["channels"] == 1
    assert clip_ref["duration_ns"] == config.end_unix_ns - config.start_unix_ns
    assert clip_ref["time_range"] == expected_time_range
    assert result["time_range"] == expected_time_range
    assert result["requested_time_range"] == _expected_requested_time_range_json(
        config.start_unix_ns,
        config.end_unix_ns,
        config.time_range_spec,
    )
    assert str(clip_ref["uri"]).startswith("file://")
    assert clip_ref["clip_id"]
    assert clip_ref["content_sha256"]
    return clip_ref


def _assert_no_metadata_ref(clip_ref) -> None:
    assert clip_ref.metadata_uri == ""
    assert clip_ref.metadata_sha256 == ""


def _assert_no_metadata_ref_json(clip_ref: JsonMapping) -> None:
    assert clip_ref["metadata_uri"] == ""
    assert clip_ref["metadata_sha256"] == ""


def _assert_time_range(time_range) -> None:
    assert time_range.start_unix_ns == _AUDIO_START_NS
    assert time_range.end_unix_ns == _AUDIO_END_NS
    assert time_range.clock == "media"
    assert time_range.uncertainty_ns == 0
    assert time_range.uncertainty_reason == ""


def _assert_profile_time_range(time_range, config: _ProfileSmokeConfig) -> None:
    assert time_range.start_unix_ns == config.start_unix_ns
    assert time_range.end_unix_ns == config.end_unix_ns
    assert time_range.clock == "media"
    assert time_range.uncertainty_ns == 0
    assert time_range.uncertainty_reason == ""


def _expected_time_range_json(start_unix_ns: int, end_unix_ns: int) -> JsonMapping:
    return {
        "start_unix_ns": start_unix_ns,
        "end_unix_ns": end_unix_ns,
        "clock": "media",
        "uncertainty_ns": 0,
        "uncertainty_reason": "",
    }


def _expected_requested_time_range_json(
    start_unix_ns: int,
    end_unix_ns: int,
    spec: str,
) -> JsonMapping:
    return {
        "start_unix_ns": start_unix_ns,
        "end_unix_ns": end_unix_ns,
        "spec": spec,
    }


def _assert_requested_time_range_duration(result: JsonMapping) -> None:
    requested = result["requested_time_range"]
    if not isinstance(requested, dict):
        raise RuntimeError("requested_time_range must be a JSON mapping")
    start_unix_ns = requested["start_unix_ns"]
    end_unix_ns = requested["end_unix_ns"]
    if not isinstance(start_unix_ns, int) or not isinstance(end_unix_ns, int):
        raise RuntimeError("requested_time_range endpoints must be integers")
    assert requested["spec"] == _RELATIVE_TIME_RANGE_SPEC
    assert end_unix_ns - start_unix_ns == _RELATIVE_EXPECTED_DURATION_NS


def _assert_wav_clip(clip_ref, *, sample_count: int) -> Path:
    clip_path = _clip_path(clip_ref)
    assert clip_path.is_file()
    assert clip_ref.content_sha256 == hashlib.sha256(clip_path.read_bytes()).hexdigest()
    with wave.open(str(clip_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == _SAMPLE_RATE
        assert wav_file.getsampwidth() == 2
        assert wav_file.getcomptype() == "NONE"
        assert wav_file.getnframes() == sample_count
        duration_ns = wav_file.getnframes() * 1_000_000_000 // wav_file.getframerate()
        assert duration_ns == sample_count * _NSEC_PER_SEC // _SAMPLE_RATE
        frames = wav_file.readframes(sample_count + 1)
    expected_payload = _pcm16le_bytes(_WINDOW_SAMPLE_VALUE, sample_count)
    assert frames == expected_payload
    return clip_path


def _assert_profile_wav_clip(clip_ref, config: _ProfileSmokeConfig) -> Path:
    clip_path = _clip_path(clip_ref)
    assert clip_ref.content_sha256 == hashlib.sha256(clip_path.read_bytes()).hexdigest()
    _assert_profile_wav_path(clip_path, config)
    return clip_path


def _assert_wav_clip_json(clip_ref: JsonMapping, *, sample_count: int) -> Path:
    clip_path = _clip_path_json(clip_ref)
    assert clip_path.is_file()
    assert clip_ref["content_sha256"] == hashlib.sha256(clip_path.read_bytes()).hexdigest()
    with wave.open(str(clip_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == _SAMPLE_RATE
        assert wav_file.getsampwidth() == 2
        assert wav_file.getcomptype() == "NONE"
        assert wav_file.getnframes() == sample_count
        duration_ns = wav_file.getnframes() * 1_000_000_000 // wav_file.getframerate()
        assert duration_ns == sample_count * _NSEC_PER_SEC // _SAMPLE_RATE
        frames = wav_file.readframes(sample_count + 1)
    expected_payload = _pcm16le_bytes(_WINDOW_SAMPLE_VALUE, sample_count)
    assert frames == expected_payload
    return clip_path


def _assert_profile_wav_clip_json(
    clip_ref: JsonMapping,
    config: _ProfileSmokeConfig,
) -> Path:
    clip_path = _clip_path_json(clip_ref)
    assert clip_ref["content_sha256"] == hashlib.sha256(clip_path.read_bytes()).hexdigest()
    _assert_profile_wav_path(clip_path, config)
    return clip_path


def _assert_profile_wav_path(clip_path: Path, config: _ProfileSmokeConfig) -> None:
    assert clip_path.is_file()
    with wave.open(str(clip_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == _SAMPLE_RATE
        assert wav_file.getsampwidth() == 2
        assert wav_file.getcomptype() == "NONE"
        assert wav_file.getnframes() == config.sample_count
        duration_ns = wav_file.getnframes() * _NSEC_PER_SEC // wav_file.getframerate()
        assert duration_ns == config.end_unix_ns - config.start_unix_ns
        frames = wav_file.readframes(config.sample_count + 1)
    expected_payload = _pcm16le_bytes(_PROFILE_ARCHIVE_SAMPLE_VALUE, config.sample_count)
    assert frames == expected_payload


def _assert_archive_metadata(clip_ref, clip_path: Path, config: _SmokeConfig) -> None:
    metadata_path = Path(str(clip_path) + ".metadata.json")
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    assert clip_ref.metadata_uri == "file://" + str(metadata_path)
    assert clip_ref.metadata_sha256 == hashlib.sha256(metadata_bytes).hexdigest()
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
    assert "metadata_uri" not in metadata["audio_clip_ref"]
    assert "content_sha256" not in metadata["audio_clip_ref"]
    assert "metadata_sha256" not in metadata["audio_clip_ref"]
    assert metadata["audio_clip_ref"]["codec"] == "pcm_s16le"
    assert metadata["audio_clip_ref"]["container"] == "wav"
    assert metadata["audio_clip_ref"]["payload_format"] == "audio/wav"
    assert metadata["audio_clip_ref"]["sample_rate"] == _SAMPLE_RATE
    assert metadata["audio_clip_ref"]["channels"] == 1
    assert metadata["audio_clip_ref"]["duration_ns"] == _EXPECTED_DURATION_NS
    assert clip_path.parent == config.archive_dir


def _assert_profile_archive_metadata(
    clip_ref,
    clip_path: Path,
    config: _ProfileSmokeConfig,
) -> None:
    metadata_path = Path(str(clip_path) + ".metadata.json")
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    assert clip_ref.metadata_uri == "file://" + str(metadata_path)
    assert clip_ref.metadata_sha256 == hashlib.sha256(metadata_bytes).hexdigest()
    assert metadata["schema"] == "fluent_audio.archive_metadata.v1"
    assert metadata["operation"] == "archive_audio_window"
    assert metadata["reason"] == _ARCHIVE_REASON
    assert metadata["related_artifact_ids"] == [_ARCHIVE_ARTIFACT_ID]
    assert metadata["source_id"] == _SOURCE_ID
    assert metadata["stream_id"] == _WINDOW_STREAM_ID
    assert metadata["window_id"] == "so101_voice_frontend_mic_archive"
    assert metadata["window_epoch"] == 1
    assert metadata["audio_scope"] == "mic"
    assert metadata["time_range"] == _expected_time_range_json(
        config.start_unix_ns,
        config.end_unix_ns,
    )
    assert metadata["audio_clip_ref"]["clip_id"] == clip_ref.clip_id
    assert metadata["audio_clip_ref"]["uri"] == clip_ref.uri
    assert "metadata_uri" not in metadata["audio_clip_ref"]
    assert "content_sha256" not in metadata["audio_clip_ref"]
    assert "metadata_sha256" not in metadata["audio_clip_ref"]
    assert metadata["audio_clip_ref"]["codec"] == "pcm_s16le"
    assert metadata["audio_clip_ref"]["container"] == "wav"
    assert metadata["audio_clip_ref"]["payload_format"] == "audio/wav"
    assert metadata["audio_clip_ref"]["sample_rate"] == _SAMPLE_RATE
    assert metadata["audio_clip_ref"]["channels"] == 1
    assert metadata["audio_clip_ref"]["duration_ns"] == config.end_unix_ns - config.start_unix_ns


def _assert_archive_metadata_json(
    clip_ref: JsonMapping,
    clip_path: Path,
    config: _SmokeConfig,
    *,
    start_unix_ns: int,
    end_unix_ns: int,
    expected_duration_ns: int,
) -> None:
    metadata_path = Path(str(clip_path) + ".metadata.json")
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    assert clip_ref["metadata_uri"] == "file://" + str(metadata_path)
    assert clip_ref["metadata_sha256"] == hashlib.sha256(metadata_bytes).hexdigest()
    assert metadata["schema"] == "fluent_audio.archive_metadata.v1"
    assert metadata["operation"] == "archive_audio_window"
    assert metadata["reason"] == _ARCHIVE_REASON
    assert metadata["related_artifact_ids"] == [_ARCHIVE_ARTIFACT_ID]
    assert metadata["source_id"] == _SOURCE_ID
    assert metadata["stream_id"] == _WINDOW_STREAM_ID
    assert metadata["window_id"] == "combined_launch_audio_window"
    assert metadata["window_epoch"] == 7
    assert metadata["audio_scope"] == "mic"
    assert metadata["time_range"] == _expected_time_range_json(start_unix_ns, end_unix_ns)
    assert metadata["audio_clip_ref"]["clip_id"] == clip_ref["clip_id"]
    assert metadata["audio_clip_ref"]["uri"] == clip_ref["uri"]
    assert "metadata_uri" not in metadata["audio_clip_ref"]
    assert "content_sha256" not in metadata["audio_clip_ref"]
    assert "metadata_sha256" not in metadata["audio_clip_ref"]
    assert metadata["audio_clip_ref"]["codec"] == "pcm_s16le"
    assert metadata["audio_clip_ref"]["container"] == "wav"
    assert metadata["audio_clip_ref"]["payload_format"] == "audio/wav"
    assert metadata["audio_clip_ref"]["sample_rate"] == _SAMPLE_RATE
    assert metadata["audio_clip_ref"]["channels"] == 1
    assert metadata["audio_clip_ref"]["duration_ns"] == expected_duration_ns
    assert clip_path.parent == config.archive_dir


def _assert_profile_archive_metadata_json(
    clip_ref: JsonMapping,
    clip_path: Path,
    config: _ProfileSmokeConfig,
) -> None:
    metadata_path = Path(str(clip_path) + ".metadata.json")
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    assert clip_ref["metadata_uri"] == "file://" + str(metadata_path)
    assert clip_ref["metadata_sha256"] == hashlib.sha256(metadata_bytes).hexdigest()
    assert metadata["schema"] == "fluent_audio.archive_metadata.v1"
    assert metadata["operation"] == "archive_audio_window"
    assert metadata["reason"] == _ARCHIVE_REASON
    assert metadata["related_artifact_ids"] == [_ARCHIVE_ARTIFACT_ID]
    assert metadata["source_id"] == _SOURCE_ID
    assert metadata["stream_id"] == _WINDOW_STREAM_ID
    assert metadata["window_id"] == "so101_voice_frontend_mic_archive"
    assert metadata["window_epoch"] == 1
    assert metadata["audio_scope"] == "mic"
    assert metadata["time_range"] == _expected_time_range_json(
        config.start_unix_ns,
        config.end_unix_ns,
    )
    assert metadata["audio_clip_ref"]["clip_id"] == clip_ref["clip_id"]
    assert metadata["audio_clip_ref"]["uri"] == clip_ref["uri"]
    assert "metadata_uri" not in metadata["audio_clip_ref"]
    assert "content_sha256" not in metadata["audio_clip_ref"]
    assert "metadata_sha256" not in metadata["audio_clip_ref"]
    assert metadata["audio_clip_ref"]["codec"] == "pcm_s16le"
    assert metadata["audio_clip_ref"]["container"] == "wav"
    assert metadata["audio_clip_ref"]["payload_format"] == "audio/wav"
    assert metadata["audio_clip_ref"]["sample_rate"] == _SAMPLE_RATE
    assert metadata["audio_clip_ref"]["channels"] == 1
    assert metadata["audio_clip_ref"]["duration_ns"] == config.end_unix_ns - config.start_unix_ns


def _clip_path(clip_ref) -> Path:
    if not clip_ref.uri.startswith("file://"):
        raise RuntimeError("audio_clip_ref.uri must be a file URI")
    return Path(clip_ref.uri.removeprefix("file://"))


def _clip_path_json(clip_ref: JsonMapping) -> Path:
    uri = str(clip_ref["uri"])
    if not uri.startswith("file://"):
        raise RuntimeError("audio_clip_ref.uri must be a file URI")
    return Path(uri.removeprefix("file://"))


def _ensure_launch_running(process: subprocess.Popen[str]) -> None:
    return_code = process.poll()
    if return_code is None:
        return
    raise RuntimeError(f"ros2 launch exited before smoke completed: {return_code}")


def _load_real_asr_smoke_input() -> _RealAsrSmokeInput:
    worker_path = _required_executable_file_env(_REAL_ASR_WORKER_ENV)
    model_path = _required_file_env(_REAL_ASR_MODEL_PATH_ENV)
    audio_path = _required_file_env(_REAL_ASR_AUDIO_F32_PATH_ENV)
    _validate_real_asr_sample_rate()
    audio_bytes = audio_path.read_bytes()
    if not audio_bytes:
        raise ValueError(f"{_REAL_ASR_AUDIO_F32_PATH_ENV} must point to a non-empty file")
    if len(audio_bytes) % 4 != 0:
        raise ValueError(
            f"{_REAL_ASR_AUDIO_F32_PATH_ENV} must be raw float32le; "
            f"byte length must be divisible by 4: {audio_path}"
        )
    samples = tuple(sample for (sample,) in struct.iter_unpack("<f", audio_bytes))
    if not all(math.isfinite(sample) for sample in samples):
        raise ValueError(
            f"{_REAL_ASR_AUDIO_F32_PATH_ENV} contains non-finite float32 samples"
        )
    if any(sample < -1.0 or sample > 1.0 for sample in samples):
        raise ValueError(
            f"{_REAL_ASR_AUDIO_F32_PATH_ENV} must contain normalized float32 samples "
            "within [-1.0, 1.0]"
        )
    return _RealAsrSmokeInput(
        worker_command=str(worker_path),
        model_path=model_path,
        audio_bytes=audio_bytes,
        expected_text=_real_asr_expected_text(),
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise AssertionError(f"{name} is required when {_REAL_ASR_SMOKE_ENV}=1")
    return value


def _required_file_env(name: str) -> Path:
    value = _required_env(name)
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise AssertionError(f"{name} must point to an existing file: {value}") from exc
    if not resolved.is_file():
        raise AssertionError(f"{name} must point to a file: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise AssertionError(f"{name} must point to a readable file: {resolved}")
    return resolved


def _required_executable_file_env(name: str) -> Path:
    resolved = _required_file_env(name)
    if not os.access(resolved, os.X_OK):
        raise AssertionError(f"{name} must point to an executable file: {resolved}")
    return resolved


def _validate_real_asr_sample_rate() -> None:
    value = os.environ.get(_REAL_ASR_SAMPLE_RATE_ENV, "").strip()
    if not value:
        return
    if value != str(_SAMPLE_RATE):
        raise ValueError(
            f"{_REAL_ASR_SAMPLE_RATE_ENV} must be {_SAMPLE_RATE} when set; got {value!r}"
        )


def _real_asr_expected_text() -> str | None:
    value = os.environ.get(_REAL_ASR_EXPECTED_TEXT_ENV)
    if value is None:
        return None
    expected_text = value.strip()
    if not expected_text:
        raise ValueError(f"{_REAL_ASR_EXPECTED_TEXT_ENV} must not be empty when set")
    return expected_text


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
