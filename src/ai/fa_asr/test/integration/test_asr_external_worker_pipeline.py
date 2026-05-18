from pathlib import Path
import sys

import numpy as np
import pytest

from fa_asr_py.backends.base import AsrRequest
from fa_asr_py.backends.factory import AsrBackendSettings, build_asr_backend


def _settings(
    tmp_path: Path,
    *,
    model_path: Path,
    output_text_path: str = "",
) -> AsrBackendSettings:
    worker = Path(__file__).parents[1] / "fixtures" / "fake_asr_worker.py"
    args = [
        str(worker),
        "--audio",
        "{audio}",
        "--model",
        "{model}",
        "--language",
        "{language}",
        "--sample-rate",
        "{sample_rate}",
        "--expected-sample",
        "0.125",
    ]
    if output_text_path:
        args.extend(["--output", "{output}"])
    return AsrBackendSettings(
        name="local_command",
        command=sys.executable,
        model="",
        model_path=str(model_path),
        openai_realtime_api_key_env="",
        openai_transcriptions_api_key_env="",
        language="ja",
        args=tuple(args),
        health_args=(),
        timeout_sec=1.0,
        working_directory="",
        output_text_path=output_text_path,
        workspace_dir=tmp_path / "workspace",
        cleanup_audio_files=True,
    )


def _request() -> AsrRequest:
    samples = np.full(1600, 0.125, dtype=np.float32)
    return AsrRequest(
        session_id="session-1",
        user_turn_id=7,
        samples=samples,
        sample_rate=16000,
    )


def test_external_worker_pipeline_reads_transcript_from_stdout(tmp_path: Path) -> None:
    model_path = tmp_path / "model.txt"
    model_path.write_text("こんにちは", encoding="utf-8")
    backend = build_asr_backend(_settings(tmp_path, model_path=model_path))

    transcript = backend.transcribe(_request())

    assert transcript == "こんにちは"
    assert list((tmp_path / "workspace").iterdir()) == []


def test_external_worker_pipeline_reads_transcript_from_output_file(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.txt"
    model_path.write_text("検証テキスト", encoding="utf-8")
    backend = build_asr_backend(
        _settings(
            tmp_path,
            model_path=model_path,
            output_text_path="transcript_{user_turn_id}.txt",
        )
    )

    transcript = backend.transcribe(_request())

    assert transcript == "検証テキスト"
    assert (tmp_path / "workspace" / "transcript_7.txt").read_text(
        encoding="utf-8"
    ) == "検証テキスト"


def test_external_worker_pipeline_rejects_empty_transcript(tmp_path: Path) -> None:
    model_path = tmp_path / "model.txt"
    model_path.write_text("", encoding="utf-8")
    backend = build_asr_backend(_settings(tmp_path, model_path=model_path))

    with pytest.raises(RuntimeError, match="empty transcript"):
        backend.transcribe(_request())
