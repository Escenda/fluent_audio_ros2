from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
from typing import TypeAlias
import uuid

import pytest
import yaml


_SMOKE_ENV = "FLUENT_AUDIO_FILE_JA_REAL_ASR_SMOKE"
_PCM_PATH_ENV = "FLUENT_AUDIO_FILE_JA_PCM_PATH"
_VAD_MODEL_DIR_ENV = "FLUENT_AUDIO_VAD_MODEL_DIR"
_VAD_PROVIDER_ENV = "FLUENT_AUDIO_VAD_PROVIDER"
_VAD_WORKER_ENV = "FLUENT_AUDIO_VAD_WORKER"
_ASR_WORKER_ENV = "FLUENT_AUDIO_NEMO_OFFLINE_TRANSCRIBE_WORKER"
_ASR_MODEL_ENV = "FLUENT_AUDIO_NEMO_OFFLINE_TRANSCRIBE_MODEL_PATH"
_EXPECTED_TEXT_ENV = "FLUENT_AUDIO_FILE_JA_EXPECTED_TEXT"

_SOURCE_ID = "file_ja"
_STREAM_ID = "audio/stream/file_ja/high_pass"
_TURN_CONTEXT_TOPIC = "conversation/file_ja/turn_context"
_VAD_STATE_TOPIC = "voice/file_ja/vad_state"
_ASR_RESULT_TOPIC = "voice/file_ja/asr/result"
_ASR_EVENT_TOPIC = "voice/asr/event"
_NSEC_PER_SEC = 1_000_000_000
_PCM16_BYTES_PER_SAMPLE = 2
_SOURCE_SAMPLE_RATE = 48_000
_TRAILING_SILENCE_SEC = 2
_OBSERVATION_TIMEOUT_SEC = 180.0
_ASR_READY_TIMEOUT_SEC = 180.0
_TEST_VAD_TIMEOUT_SEC = 3.0
_TEST_VAD_PROVIDER = "cpu"

_TEST_VAD_WORKER = """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct


_SPEECH_THRESHOLD = 1.0e-4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--window-samples", type=int, required=True)
    args = parser.parse_args()

    if args.provider != "cpu":
        raise RuntimeError(f"provider must be cpu: {args.provider}")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise RuntimeError(f"model path must be a directory: {model_path}")
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise RuntimeError(f"audio path must be a file: {audio_path}")
    if args.sample_rate != 16000:
        raise RuntimeError(f"sample rate must be 16000: {args.sample_rate}")
    if args.window_samples <= 0:
        raise RuntimeError("window samples must be positive")

    payload = audio_path.read_bytes()
    if not payload:
        raise RuntimeError("audio payload must be non-empty")
    if len(payload) % 4 != 0:
        raise RuntimeError("audio payload must be raw float32le")
    samples = [sample for (sample,) in struct.iter_unpack("<f", payload)]
    if not samples:
        raise RuntimeError("audio payload must contain samples")
    max_abs = 0.0
    for sample in samples:
        if not math.isfinite(sample):
            raise RuntimeError("audio samples must be finite")
        if abs(sample) > 1.0:
            raise RuntimeError("audio samples must be normalized to [-1.0, 1.0]")
        max_abs = max(max_abs, abs(sample))

    print("0.75" if max_abs > _SPEECH_THRESHOLD else "0.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

YamlScalar: TypeAlias = str | int | float | bool | None
YamlMapping: TypeAlias = dict[str, "YamlValue"]
YamlSequence: TypeAlias = list["YamlValue"]
YamlValue: TypeAlias = YamlScalar | YamlMapping | YamlSequence


@dataclass(frozen=True)
class _ExecutableEnv:
    env_name: str
    configured_value: str
    resolved_path: Path


@dataclass(frozen=True)
class _SmokeInput:
    source_pcm_path: Path
    conditioned_pcm_path: Path
    downstream_config_path: Path
    source_config_path: Path
    vad_model_dir: Path
    vad_worker_path: Path
    asr_worker: _ExecutableEnv
    asr_model_path: Path
    expected_text: str | None


def test_file_ja_voice_frontend_runs_real_nemo_offline_asr_when_enabled(
    tmp_path: Path,
) -> None:
    if os.environ.get(_SMOKE_ENV) != "1":
        pytest.skip(f"{_SMOKE_ENV}=1 is required for file_ja real ASR smoke")

    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise AssertionError("ros2 executable is required when file_ja real ASR smoke is enabled")

    smoke_input = _load_smoke_input(tmp_path)
    suffix = "file_ja_real_asr_" + uuid.uuid4().hex
    domain_id = _isolated_ros_domain_id(suffix)
    previous_domain_id = os.environ.get("ROS_DOMAIN_ID")
    os.environ["ROS_DOMAIN_ID"] = domain_id
    try:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        from fa_interfaces.msg import AsrEvent, AsrResult, TurnContext, VadState

        context = Context()
        rclpy.init(context=context)
        node = rclpy.create_node(f"{suffix}_client", context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        vad_states: list[VadState] = []
        asr_results: list[AsrResult] = []
        asr_events: list[AsrEvent] = []
        vad_state_sub = node.create_subscription(
            VadState,
            _VAD_STATE_TOPIC,
            vad_states.append,
            _best_effort_qos(QoSProfile, ReliabilityPolicy),
        )
        asr_result_sub = node.create_subscription(
            AsrResult,
            _ASR_RESULT_TOPIC,
            asr_results.append,
            10,
        )
        asr_event_sub = node.create_subscription(
            AsrEvent,
            _ASR_EVENT_TOPIC,
            asr_events.append,
            50,
        )
        turn_context_pub = node.create_publisher(
            TurnContext,
            _TURN_CONTEXT_TOPIC,
            _reliable_qos(QoSProfile, ReliabilityPolicy),
        )
        downstream_process = _start_profile_launch_process(
            ros2,
            smoke_input.downstream_config_path,
            smoke_input,
            domain_id=domain_id,
            fa_in_enabled=False,
        )
        source_process: subprocess.Popen[str] | None = None
        try:
            session_id = f"file-ja-real-asr-smoke-{suffix}"
            user_turn_id = 1
            _wait_for_downstream_asr_ready(
                executor,
                downstream_process,
                turn_context_pub,
                TurnContext,
                AsrEvent,
                AsrResult,
                asr_results,
                asr_events,
                session_id=session_id,
                user_turn_id=user_turn_id,
            )
            source_process = _start_profile_launch_process(
                ros2,
                smoke_input.source_config_path,
                smoke_input,
                domain_id=domain_id,
                fa_in_enabled=True,
            )
            final_result = _observe_profile_until_final_result(
                executor,
                downstream_process,
                source_process,
                turn_context_pub,
                TurnContext,
                AsrEvent,
                AsrResult,
                vad_states,
                asr_results,
                asr_events,
                session_id=session_id,
                user_turn_id=user_turn_id,
            )
            assert final_result.session_id == session_id
            assert final_result.user_turn_id > 0
            transcript = final_result.text.strip()
            assert transcript
            if smoke_input.expected_text:
                assert smoke_input.expected_text.strip() in transcript
        except Exception as exc:
            downstream_stdout = _stop_process(downstream_process)
            source_stdout = ""
            if source_process is not None:
                source_stdout = _stop_process(source_process)
            raise AssertionError(
                f"{exc}\n\n"
                f"downstream ros2 launch output:\n{downstream_stdout}\n\n"
                f"source ros2 launch output:\n{source_stdout}"
            ) from exc
        finally:
            if source_process is not None and source_process.poll() is None:
                _stop_process(source_process)
            if downstream_process.poll() is None:
                _stop_process(downstream_process)
            node.destroy_publisher(turn_context_pub)
            node.destroy_subscription(asr_event_sub)
            node.destroy_subscription(asr_result_sub)
            node.destroy_subscription(vad_state_sub)
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok(context=context):
                rclpy.shutdown(context=context)
    finally:
        if previous_domain_id is None:
            os.environ.pop("ROS_DOMAIN_ID", None)
        else:
            os.environ["ROS_DOMAIN_ID"] = previous_domain_id


def _load_smoke_input(tmp_path: Path) -> _SmokeInput:
    source_pcm_path = _required_file_path(_PCM_PATH_ENV)
    _validate_raw_pcm16le_mono_file(source_pcm_path, _PCM_PATH_ENV)
    conditioned_pcm_path = _condition_pcm_fixture_with_trailing_silence(
        source_pcm_path,
        tmp_path,
    )
    downstream_config_path, source_config_path = _write_split_profile_configs(tmp_path)
    vad_model_dir, vad_worker_path = _write_test_vad_runtime(tmp_path)
    asr_worker = _required_executable_env(_ASR_WORKER_ENV)
    asr_model_path = _required_file_path(_ASR_MODEL_ENV)
    if asr_model_path.suffix != ".nemo":
        raise AssertionError(f"{_ASR_MODEL_ENV} must point to a local .nemo file: {asr_model_path}")
    expected_text = os.environ.get(_EXPECTED_TEXT_ENV)
    if expected_text is not None and not expected_text.strip():
        raise AssertionError(f"{_EXPECTED_TEXT_ENV} must be non-empty when set")
    return _SmokeInput(
        source_pcm_path=source_pcm_path,
        conditioned_pcm_path=conditioned_pcm_path,
        downstream_config_path=downstream_config_path,
        source_config_path=source_config_path,
        vad_model_dir=vad_model_dir,
        vad_worker_path=vad_worker_path,
        asr_worker=asr_worker,
        asr_model_path=asr_model_path,
        expected_text=expected_text,
    )


def _condition_pcm_fixture_with_trailing_silence(source_pcm_path: Path, tmp_path: Path) -> Path:
    conditioned_pcm_path = tmp_path / "file_ja_voice_frontend_input_with_trailing_silence.pcm"
    source_bytes = source_pcm_path.read_bytes()
    trailing_silence = b"\x00" * (
        _SOURCE_SAMPLE_RATE * _TRAILING_SILENCE_SEC * _PCM16_BYTES_PER_SAMPLE
    )
    conditioned_pcm_path.write_bytes(source_bytes + trailing_silence)
    return conditioned_pcm_path


def _write_test_vad_runtime(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "deterministic_test_vad_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "README.txt").write_text(
        "Deterministic VAD fixture for file_ja real ASR smoke.\n",
        encoding="utf-8",
    )
    (model_dir / "hubconf.py").write_text("def silero_vad():\n    return None\n", encoding="utf-8")
    worker_path = tmp_path / "deterministic_test_vad_worker.py"
    worker_path.write_text(_TEST_VAD_WORKER, encoding="utf-8")
    worker_path.chmod(0o755)
    return model_dir, worker_path


def _validate_raw_pcm16le_mono_file(path: Path, env_name: str) -> None:
    byte_count = path.stat().st_size
    if byte_count <= 0:
        raise AssertionError(f"{env_name} must point to a non-empty raw PCM16LE mono file: {path}")
    if byte_count % _PCM16_BYTES_PER_SAMPLE != 0:
        raise AssertionError(
            f"{env_name} byte length must align to 16-bit mono samples: "
            f"path={path} bytes={byte_count}"
        )


def _required_file_path(env_name: str) -> Path:
    value = _required_non_empty_env(env_name)
    path = Path(value).expanduser()
    if not path.is_file():
        raise AssertionError(f"{env_name} must point to an existing file: {path}")
    return path


def _required_non_empty_env(env_name: str) -> str:
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        raise AssertionError(f"{env_name} must be set for file_ja real ASR smoke")
    return value.strip()


def _required_executable_env(env_name: str) -> _ExecutableEnv:
    configured_value = _required_non_empty_env(env_name)
    resolved = _resolve_executable(configured_value)
    if resolved is None:
        raise AssertionError(f"{env_name} must resolve to an executable: {configured_value}")
    return _ExecutableEnv(
        env_name=env_name,
        configured_value=configured_value,
        resolved_path=resolved,
    )


def _resolve_executable(configured_value: str) -> Path | None:
    if "/" in configured_value:
        path = Path(configured_value).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        return None
    resolved = shutil.which(configured_value)
    if resolved is None:
        return None
    path = Path(resolved)
    if path.is_file() and os.access(path, os.X_OK):
        return path
    return None


def _write_split_profile_configs(tmp_path: Path) -> tuple[Path, Path]:
    profile = _load_source_profile()
    downstream_profile = deepcopy(profile)
    source_profile = deepcopy(profile)
    downstream_profile["groups"] = _profile_groups_except_audio_io(downstream_profile)
    source_profile["groups"] = _profile_audio_io_groups(source_profile)
    _apply_test_vad_timeout(downstream_profile)

    downstream_config_path = tmp_path / "file_ja_voice_frontend_downstream.yaml"
    source_config_path = tmp_path / "file_ja_voice_frontend_source.yaml"
    _write_yaml(downstream_config_path, downstream_profile)
    _write_yaml(source_config_path, source_profile)
    return downstream_config_path, source_config_path


def _load_source_profile() -> YamlMapping:
    profile_path = (
        Path(__file__).parents[2]
        / "config"
        / "profiles"
        / "file_ja_voice_frontend.yaml"
    )
    loaded: YamlValue = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"file_ja profile must be a YAML mapping: {profile_path}")
    return loaded


def _write_yaml(path: Path, value: YamlMapping) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _profile_groups_except_audio_io(profile: YamlMapping) -> YamlSequence:
    return [
        group
        for group in _profile_groups(profile)
        if _mapping_string(group, "id") != "audio_io"
    ]


def _profile_audio_io_groups(profile: YamlMapping) -> YamlSequence:
    audio_io_groups = [
        group
        for group in _profile_groups(profile)
        if _mapping_string(group, "id") == "audio_io"
    ]
    if len(audio_io_groups) != 1:
        raise RuntimeError(
            "file_ja profile must contain exactly one audio_io group for source split"
        )
    return audio_io_groups


def _profile_groups(profile: YamlMapping) -> YamlSequence:
    groups = profile.get("groups")
    if not isinstance(groups, list):
        raise RuntimeError("file_ja profile must define groups as a YAML sequence")
    return groups


def _apply_test_vad_timeout(profile: YamlMapping) -> None:
    for group in _profile_groups(profile):
        if not isinstance(group, dict):
            continue
        nodes = group.get("nodes")
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if _mapping_string(node, "id") != "fa_vad_file_ja":
                continue
            parameters = node.get("parameters")
            if not isinstance(parameters, dict):
                raise RuntimeError("fa_vad_file_ja must define parameters mapping")
            parameters["backend.timeout_sec"] = _TEST_VAD_TIMEOUT_SEC
            return
    raise RuntimeError("fa_vad_file_ja was not found in downstream profile split")


def _mapping_string(mapping: YamlMapping, key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"expected YAML string at key {key}")
    return value


def _start_profile_launch_process(
    ros2: str,
    config_path: Path,
    smoke_input: _SmokeInput,
    *,
    domain_id: str,
    fa_in_enabled: bool,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(
        {
            "ROS_DOMAIN_ID": domain_id,
            _PCM_PATH_ENV: str(smoke_input.conditioned_pcm_path),
            _VAD_MODEL_DIR_ENV: str(smoke_input.vad_model_dir),
            _VAD_PROVIDER_ENV: _TEST_VAD_PROVIDER,
            _VAD_WORKER_ENV: str(smoke_input.vad_worker_path),
            _ASR_WORKER_ENV: smoke_input.asr_worker.configured_value,
            _ASR_MODEL_ENV: str(smoke_input.asr_model_path),
        }
    )
    return subprocess.Popen(
        [
            ros2,
            "launch",
            "fluent_audio_system",
            "run.py",
            f"config:={config_path}",
            f"fa_in_enabled:={str(fa_in_enabled).lower()}",
            "fa_out_enabled:=false",
            f"fa_in_source_id:={_SOURCE_ID}",
            "fa_out_sink_id:=disabled",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )


def _wait_for_downstream_asr_ready(
    executor,
    downstream_process: subprocess.Popen[str],
    turn_context_pub,
    turn_context_cls,
    asr_event_cls,
    asr_result_cls,
    asr_results,
    asr_events,
    *,
    session_id: str,
    user_turn_id: int,
) -> None:
    context_msg = _active_turn_context(turn_context_cls, session_id, user_turn_id)
    deadline = time.monotonic() + _ASR_READY_TIMEOUT_SEC
    next_context_publish_at = 0.0
    while time.monotonic() < deadline:
        _ensure_launch_running(downstream_process)
        now = time.monotonic()
        if now >= next_context_publish_at:
            turn_context_pub.publish(context_msg)
            next_context_publish_at = now + 0.1
        executor.spin_once(timeout_sec=0.05)
        _raise_for_forbidden_asr_outputs(asr_result_cls, asr_event_cls, asr_results, asr_events)
        if asr_events:
            return
    raise RuntimeError("downstream fa_asr did not emit readiness ASR event before timeout")


def _observe_profile_until_final_result(
    executor,
    downstream_process: subprocess.Popen[str],
    source_process: subprocess.Popen[str],
    turn_context_pub,
    turn_context_cls,
    asr_event_cls,
    asr_result_cls,
    vad_states,
    asr_results,
    asr_events,
    *,
    session_id: str,
    user_turn_id: int,
):
    context_msg = _active_turn_context(turn_context_cls, session_id, user_turn_id)
    vad_start_seen = False
    vad_end_seen = False
    final_result = None
    deadline = time.monotonic() + _OBSERVATION_TIMEOUT_SEC
    next_context_publish_at = 0.0
    while time.monotonic() < deadline:
        _ensure_launch_running(downstream_process)
        _ensure_launch_not_failed(source_process)
        now = time.monotonic()
        if now >= next_context_publish_at:
            turn_context_pub.publish(context_msg)
            next_context_publish_at = now + 0.1
        executor.spin_once(timeout_sec=0.05)
        _raise_for_forbidden_asr_outputs(asr_result_cls, asr_event_cls, asr_results, asr_events)
        vad_start_seen = vad_start_seen or _vad_start_seen(vad_states)
        vad_end_seen = vad_end_seen or _vad_end_seen(vad_states)
        final_result = _final_asr_result(asr_result_cls, asr_results)
        if vad_start_seen and vad_end_seen and final_result is not None:
            return final_result
    raise RuntimeError(
        "file_ja real ASR smoke timed out: "
        f"vad_start={vad_start_seen} vad_end={vad_end_seen} "
        f"final_asr={final_result is not None} "
        f"vad_states={len(vad_states)} asr_results={len(asr_results)} asr_events={len(asr_events)}"
    )


def _active_turn_context(turn_context_cls, session_id: str, user_turn_id: int):
    msg = turn_context_cls()
    now_ns = time.time_ns()
    stamp_sec, stamp_nanosec = divmod(now_ns, _NSEC_PER_SEC)
    msg.timestamp.sec = stamp_sec
    msg.timestamp.nanosec = stamp_nanosec
    msg.session_id = session_id
    msg.user_turn_id = user_turn_id
    msg.active = True
    return msg


def _vad_start_seen(vad_states) -> bool:
    return any(
        state.source_id == _SOURCE_ID
        and state.stream_id == _STREAM_ID
        and state.start
        for state in vad_states
    )


def _vad_end_seen(vad_states) -> bool:
    return any(
        state.source_id == _SOURCE_ID
        and state.stream_id == _STREAM_ID
        and state.end
        for state in vad_states
    )


def _final_asr_result(asr_result_cls, asr_results):
    for result in asr_results:
        if result.status == asr_result_cls.STATUS_FINAL:
            if result.text.strip():
                return result
            raise RuntimeError("ASR final result text was empty")
    return None


def _raise_for_forbidden_asr_outputs(
    asr_result_cls,
    asr_event_cls,
    asr_results,
    asr_events,
) -> None:
    for result in asr_results:
        if result.status == asr_result_cls.STATUS_ERROR:
            raise RuntimeError(f"ASR returned STATUS_ERROR: reason={result.reason}")

    forbidden_events = _forbidden_event_codes(asr_event_cls)
    for event in asr_events:
        if event.event in forbidden_events:
            raise RuntimeError(
                "ASR emitted forbidden event: "
                f"event={event.event} reason={event.reason} error_code={event.error_code}"
            )


def _forbidden_event_codes(asr_event_cls) -> set[int]:
    names = (
        "EVENT_FAIL_CLOSED",
        "EVENT_AUDIO_FRAME_TIMELINE_APPEND_DROPPED",
        "EVENT_TIMELINE_OVERLAP_DERIVED_FAILURE",
        "EVENT_INVALID_AUDIO_FRAME_DROPPED",
    )
    codes: set[int] = set()
    for name in names:
        if hasattr(asr_event_cls, name):
            codes.add(int(getattr(asr_event_cls, name)))
    return codes


def _reliable_qos(qos_profile_cls, reliability_policy_cls):
    qos = qos_profile_cls(depth=10)
    qos.reliability = reliability_policy_cls.RELIABLE
    return qos


def _best_effort_qos(qos_profile_cls, reliability_policy_cls):
    qos = qos_profile_cls(depth=10)
    qos.reliability = reliability_policy_cls.BEST_EFFORT
    return qos


def _isolated_ros_domain_id(suffix: str) -> str:
    return str(20 + (int(suffix[-4:], 16) % 80))


def _ensure_launch_running(process: subprocess.Popen[str]) -> None:
    return_code = process.poll()
    if return_code is None:
        return
    raise RuntimeError(f"ros2 launch exited before smoke completed: {return_code}")


def _ensure_launch_not_failed(process: subprocess.Popen[str]) -> None:
    return_code = process.poll()
    if return_code is None or return_code == 0:
        return
    raise RuntimeError(f"ros2 launch exited with failure before smoke completed: {return_code}")


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
