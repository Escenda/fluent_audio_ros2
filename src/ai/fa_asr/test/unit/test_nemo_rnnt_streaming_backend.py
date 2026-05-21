import base64
import json
from pathlib import Path

import numpy as np
import pytest

from fa_asr_py.backends.base import (
    ASR_AUDIO_ENCODING_FLOAT32LE,
    AsrAudioPayload,
    AsrStreamRequest,
    asr_transcript_text,
)
from fa_asr_py.backends.factory import AsrBackendSettings, build_asr_backend
from fa_asr_py.backends.nemo_rnnt_streaming import (
    NemoRnntStreamingAsrBackend,
    load_nemo_rnnt_streaming_config,
)


def _write_worker(path: Path, *, behavior: str = "ok") -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


behavior = os.environ.get("FAKE_NEMO_WORKER_BEHAVIOR", "ok")
log_path = Path(os.environ["FAKE_NEMO_WORKER_LOG"])


def emit(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\\n")
    sys.stdout.flush()


def record(message):
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(message, separators=(",", ":")) + "\\n")


for line in sys.stdin:
    message = json.loads(line)
    record(message)
    if behavior == "malformed_json" and message["type"] == "audio":
        sys.stdout.write("{bad json\\n")
        sys.stdout.flush()
        continue
    if message["type"] == "health":
        if behavior == "offline_model":
            emit({
                "type": "health_ok",
                "model_class": "ctc",
                "cache_aware_streaming": False,
                "sample_rate_hz": message["sample_rate_hz"],
                "channels": message["channels"],
                "audio_encoding": message["audio_encoding"],
                "streaming": True,
                "final_results_only": not message["emit_partial"],
                "supports_partials": True,
                "language": message["language"],
                "chunk_size_samples": message["chunk_size_samples"],
                "max_partial_interval_ms": message["max_partial_interval_ms"],
            })
            continue
        emit({
            "type": "health_ok",
            "model_class": "rnnt",
            "cache_aware_streaming": True,
            "sample_rate_hz": message["sample_rate_hz"],
            "channels": message["channels"],
            "audio_encoding": message["audio_encoding"],
            "streaming": True,
            "final_results_only": not message["emit_partial"],
            "supports_partials": True,
            "language": message["language"],
            "chunk_size_samples": message["chunk_size_samples"],
            "max_partial_interval_ms": message["max_partial_interval_ms"],
        })
        continue
    if message["type"] == "start":
        emit({"type": "stream_started", "session_id": message["session_id"]})
        continue
    if message["type"] == "audio":
        if behavior == "result_beyond_pushed":
            emit({
                "type": "partial",
                "session_id": message["session_id"],
                "text": "too far",
                "sample_count": message["sample_count"] + 1,
            })
        elif behavior == "unsupported_result_field":
            emit({
                "type": "partial",
                "session_id": message["session_id"],
                "text": "partial",
                "sample_count": message["sample_count"],
                "confidence": 0.9,
            })
        elif behavior == "empty_transcript":
            emit({
                "type": "partial",
                "session_id": message["session_id"],
                "text": "",
                "sample_count": message["sample_count"],
            })
        else:
            emit({
                "type": "partial",
                "session_id": message["session_id"],
                "text": "partial",
                "sample_count": message["sample_count"],
            })
        emit({
            "type": "audio_accepted",
            "session_id": message["session_id"],
            "sample_count": message["sample_count"],
        })
        continue
    if message["type"] == "drain":
        emit({"type": "drained", "session_id": message["session_id"]})
        continue
    if message["type"] == "finish":
        if behavior != "missing_final":
            emit({
                "type": "final",
                "session_id": message["session_id"],
                "text": "" if behavior == "empty_final_transcript" else "final text",
                "sample_count": 4,
            })
        emit({"type": "finished", "session_id": message["session_id"]})
        break
    if message["type"] == "cancel":
        emit({"type": "cancelled", "session_id": message["session_id"]})
        break
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _model_path(tmp_path: Path) -> Path:
    path = tmp_path / "streaming.nemo"
    path.write_bytes(b"fake-nemo")
    return path


def _backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    behavior: str = "ok",
    emit_partial: bool = True,
) -> NemoRnntStreamingAsrBackend:
    worker = _write_worker(tmp_path / "worker.py", behavior=behavior)
    monkeypatch.setenv("FAKE_NEMO_WORKER_LOG", str(tmp_path / "worker.jsonl"))
    monkeypatch.setenv("FAKE_NEMO_WORKER_BEHAVIOR", behavior)
    backend = build_asr_backend(
        AsrBackendSettings(
            name="nemo_rnnt_streaming",
            command=str(worker),
            model_path=str(_model_path(tmp_path)),
            language="ja",
            timeout_sec=5.0,
            sample_rate_hz=16000,
            channels=1,
            chunk_size_samples=4,
            emit_partial=emit_partial,
            max_partial_interval_ms=300,
        )
    )
    assert isinstance(backend, NemoRnntStreamingAsrBackend)
    return backend


def _payload() -> AsrAudioPayload:
    return AsrAudioPayload.from_float32_samples(
        np.array([0.1, 0.2, -0.2, 0.0], dtype=np.float32),
        sample_rate_hz=16000,
    )


def _read_log(path: Path) -> list[dict[str, str | int | bool]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_factory_builds_nemo_rnnt_streaming_with_health_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path, monkeypatch)
    assert backend.capability.audio_encoding == ASR_AUDIO_ENCODING_FLOAT32LE
    assert backend.capability.sample_rate_hz == 16000
    assert backend.capability.channels == 1
    assert backend.capability.streaming is True
    assert backend.capability.final_results_only is False


def test_config_requires_existing_local_nemo_model(tmp_path: Path) -> None:
    worker = _write_worker(tmp_path / "worker.py")
    with pytest.raises(RuntimeError, match="backend.model_path does not exist"):
        load_nemo_rnnt_streaming_config(
            command=str(worker),
            model_path_value=str(tmp_path / "missing.nemo"),
            language="ja",
            timeout_sec=1.0,
            working_directory_value="",
            sample_rate_hz=16000,
            channels=1,
            chunk_size_samples=4,
            chunk_ms=0,
            emit_partial=True,
            max_partial_interval_ms=300,
        )


def test_health_rejects_non_rnnt_or_non_cache_aware_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="non-RNNT model"):
        _backend(tmp_path, monkeypatch, behavior="offline_model")


def test_start_sends_config_before_audio_and_push_sends_base64_float32le(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path, monkeypatch)
    session = backend.start_stream(AsrStreamRequest(session_id="s1", user_turn_id=7))
    results = session.push_audio(_payload())
    assert len(results) == 1
    assert results[0].is_final is False
    assert asr_transcript_text(results[0].transcript) == "partial"

    messages = _read_log(tmp_path / "worker.jsonl")
    assert [message["type"] for message in messages[:3]] == ["health", "start", "audio"]
    start = messages[1]
    assert start["model_path"] == str(tmp_path / "streaming.nemo")
    assert start["language"] == "ja"
    assert start["sample_rate_hz"] == 16000
    assert start["chunk_size_samples"] == 4

    audio = messages[2]
    assert audio["encoding"] == "base64_float32le"
    assert audio["sample_count"] == 4
    assert base64.b64decode(str(audio["data"])) == _payload().data


def test_finish_requires_final_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(tmp_path, monkeypatch, behavior="missing_final")
    session = backend.start_stream(AsrStreamRequest(session_id="s1", user_turn_id=7))
    session.push_audio(_payload())
    with pytest.raises(RuntimeError, match="finish did not return a final result"):
        session.finish()


def test_finish_maps_final_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(tmp_path, monkeypatch)
    session = backend.start_stream(AsrStreamRequest(session_id="s1", user_turn_id=7))
    session.push_audio(_payload())
    results = session.finish()
    assert len(results) == 1
    assert results[0].is_final is True
    assert asr_transcript_text(results[0].transcript) == "final text"


def test_finish_maps_empty_final_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path, monkeypatch, behavior="empty_final_transcript")
    session = backend.start_stream(AsrStreamRequest(session_id="s1", user_turn_id=7))
    session.push_audio(_payload())
    results = session.finish()
    assert len(results) == 1
    assert results[0].is_final is True
    assert asr_transcript_text(results[0].transcript) == ""


def test_cancel_sends_cancel_without_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(tmp_path, monkeypatch)
    session = backend.start_stream(AsrStreamRequest(session_id="s1", user_turn_id=7))
    session.cancel()
    messages = _read_log(tmp_path / "worker.jsonl")
    assert [message["type"] for message in messages] == ["health", "start", "cancel"]


@pytest.mark.parametrize(
    ("behavior", "message"),
    (
        ("malformed_json", "malformed JSONL"),
        ("result_beyond_pushed", "exceeds pushed audio sample count"),
        ("unsupported_result_field", "unsupported fields"),
        ("empty_transcript", "empty transcript"),
    ),
)
def test_worker_protocol_failures_are_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    message: str,
) -> None:
    backend = _backend(tmp_path, monkeypatch, behavior=behavior)
    session = backend.start_stream(AsrStreamRequest(session_id="s1", user_turn_id=7))
    with pytest.raises(RuntimeError, match=message):
        session.push_audio(_payload())


def test_emit_partial_false_reports_final_only_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path, monkeypatch, emit_partial=False)
    assert backend.capability.final_results_only is True
