import json
import os
from pathlib import Path

import numpy as np
import pytest

from fa_asr_py.backends.base import (
    ASR_AUDIO_ENCODING_FLOAT32LE,
    AsrAudioPayload,
    AsrRequest,
    asr_transcript_text,
)
from fa_asr_py.backends.factory import AsrBackendSettings, build_asr_backend
from fa_asr_py.backends.nemo_offline_transcribe import (
    NemoOfflineTranscribeAsrBackend,
    load_nemo_offline_transcribe_config,
)


def _write_worker(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import struct
import sys


def _flag_value(flag):
    for index, value in enumerate(sys.argv):
        if value == flag and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    raise RuntimeError(f"missing flag: {flag}")


log_path = Path(os.environ["FAKE_NEMO_OFFLINE_LOG"])
with log_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:], separators=(",", ":")) + "\\n")

if sys.argv[1] == "health":
    if os.environ.get("FAKE_NEMO_OFFLINE_HEALTH_FAIL") == "1":
        print("forced health failure", file=sys.stderr)
        raise SystemExit(7)
    if Path(_flag_value("--model")).suffix != ".nemo":
        raise RuntimeError("model must be .nemo")
    if int(_flag_value("--sample-rate")) <= 0:
        raise RuntimeError("sample rate must be positive")
    if int(_flag_value("--channels")) != 1:
        raise RuntimeError("channels must be mono")
    print("health ok")
    raise SystemExit(0)

if sys.argv[1] != "transcribe":
    raise RuntimeError("unsupported mode")
audio_path = Path(_flag_value("--audio"))
model_path = Path(_flag_value("--model"))
result_format = _flag_value("--result-format")
sample_rate = int(_flag_value("--sample-rate"))
channels = int(_flag_value("--channels"))
if not model_path.is_file():
    raise RuntimeError("model missing")
if sample_rate != 16000:
    raise RuntimeError("unexpected sample rate")
if channels != 1:
    raise RuntimeError("unexpected channel count")
payload = audio_path.read_bytes()
if not payload or len(payload) % 4 != 0:
    raise RuntimeError("expected raw float32le audio")
samples = [sample for (sample,) in struct.iter_unpack("<f", payload)]
if samples != [0.125, -0.25]:
    raise RuntimeError(f"unexpected samples: {samples}")
if result_format == "segments_json_v1":
    print(json.dumps({
        "result_format": "segments_json_v1",
        "segments": [{"start_sample": 0, "end_sample": len(samples), "text": "segment text"}],
    }, separators=(",", ":")))
else:
    print("plain transcript")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _model_path(tmp_path: Path) -> Path:
    path = tmp_path / "parakeet.nemo"
    path.write_bytes(b"fake nemo")
    return path


def _settings(
    tmp_path: Path,
    *,
    result_format: str = "plain_text",
) -> AsrBackendSettings:
    return AsrBackendSettings(
        name="nemo_offline_transcribe",
        command=str(_write_worker(tmp_path / "worker.py")),
        model_path=str(_model_path(tmp_path)),
        language="ja",
        timeout_sec=5.0,
        workspace_dir=tmp_path / "work",
        cleanup_audio_files=True,
        result_format=result_format,
        sample_rate_hz=16000,
        channels=1,
    )


def _request(*, sample_rate_hz: int = 16000) -> AsrRequest:
    return AsrRequest(
        session_id="session",
        user_turn_id=3,
        payload=AsrAudioPayload.from_float32_samples(
            np.array([0.125, -0.25], dtype=np.float32),
            sample_rate_hz=sample_rate_hz,
        ),
    )


def test_factory_builds_non_streaming_backend_and_runs_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_NEMO_OFFLINE_LOG", str(tmp_path / "worker.jsonl"))

    backend = build_asr_backend(_settings(tmp_path))

    assert isinstance(backend, NemoOfflineTranscribeAsrBackend)
    assert backend.capability.audio_encoding == ASR_AUDIO_ENCODING_FLOAT32LE
    assert backend.capability.sample_rate_hz == 16000
    assert backend.capability.channels == 1
    assert backend.capability.streaming is False
    assert backend.capability.final_results_only is True
    assert not hasattr(backend, "start_stream")
    first_call = json.loads((tmp_path / "worker.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_call[:2] == ["health", "--model"]


def test_transcribe_serializes_float32le_and_parses_plain_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_NEMO_OFFLINE_LOG", str(tmp_path / "worker.jsonl"))
    backend = build_asr_backend(_settings(tmp_path))

    transcript = backend.transcribe(_request())

    assert asr_transcript_text(transcript) == "plain transcript"
    calls = [
        json.loads(line)
        for line in (tmp_path / "worker.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert calls[1][0] == "transcribe"
    assert "--channels" in calls[1]


def test_transcribe_parses_segments_json_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_NEMO_OFFLINE_LOG", str(tmp_path / "worker.jsonl"))
    backend = build_asr_backend(_settings(tmp_path, result_format="segments_json_v1"))

    transcript = backend.transcribe(_request())

    assert asr_transcript_text(transcript) == "segment text"
    assert transcript.segments[0].start_sample == 0
    assert transcript.segments[0].end_sample == 2


def test_config_requires_local_nemo_suffix(tmp_path: Path) -> None:
    command = _write_worker(tmp_path / "worker.py")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"not nemo")

    with pytest.raises(RuntimeError, match="local .nemo file"):
        load_nemo_offline_transcribe_config(
            command=str(command),
            model_path_value=str(model_path),
            language="ja",
            timeout_sec=1.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
            sample_rate_hz=16000,
            channels=1,
        )


def test_config_rejects_non_mono_channels(tmp_path: Path) -> None:
    command = _write_worker(tmp_path / "worker.py")

    with pytest.raises(RuntimeError, match="backend.channels must be 1"):
        load_nemo_offline_transcribe_config(
            command=str(command),
            model_path_value=str(_model_path(tmp_path)),
            language="ja",
            timeout_sec=1.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
            sample_rate_hz=16000,
            channels=2,
        )


def test_health_failure_fails_backend_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_NEMO_OFFLINE_LOG", str(tmp_path / "worker.jsonl"))
    monkeypatch.setenv("FAKE_NEMO_OFFLINE_HEALTH_FAIL", "1")

    with pytest.raises(RuntimeError, match="health check failed"):
        build_asr_backend(_settings(tmp_path))


def test_backend_rejects_request_sample_rate_mismatch_before_worker_transcribe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_NEMO_OFFLINE_LOG", str(tmp_path / "worker.jsonl"))
    backend = build_asr_backend(_settings(tmp_path))

    with pytest.raises(ValueError, match="sample_rate_hz must be 16000"):
        backend.transcribe(_request(sample_rate_hz=8000))

    calls = [
        json.loads(line)
        for line in (tmp_path / "worker.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 1
