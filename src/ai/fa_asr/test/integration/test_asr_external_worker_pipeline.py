"""Legacy optional local_command backend tests.

These tests exercise the external-process backend contract only. They are not
validation for the standard parakeet_multilingual_buffered ASR path.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

from fa_asr_py.backends.base import AsrAudioPayload, AsrRequest, asr_transcript_text
from fa_asr_py.backends.factory import AsrBackendSettings, build_asr_backend


def _settings(
    tmp_path: Path,
    *,
    model_path: Path,
    output_text_path: str = "",
    result_format: str = "plain_text",
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
        result_format=result_format,
    )


def _request() -> AsrRequest:
    samples = np.full(1600, 0.125, dtype=np.float32)
    return AsrRequest(
        session_id="session-1",
        user_turn_id=7,
        payload=AsrAudioPayload.from_float32_samples(samples, sample_rate_hz=16000),
    )


def test_external_worker_pipeline_reads_transcript_from_stdout(tmp_path: Path) -> None:
    model_path = tmp_path / "model.txt"
    model_path.write_text("こんにちは", encoding="utf-8")
    backend = build_asr_backend(_settings(tmp_path, model_path=model_path))

    transcript = backend.transcribe(_request())

    assert asr_transcript_text(transcript) == "こんにちは"
    assert transcript.segments[0].start_sample == 0
    assert transcript.segments[0].end_sample == 1600
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

    assert asr_transcript_text(transcript) == "検証テキスト"
    assert (tmp_path / "workspace" / "transcript_7.txt").read_text(
        encoding="utf-8"
    ) == "検証テキスト"


def test_external_worker_pipeline_rejects_empty_transcript(tmp_path: Path) -> None:
    model_path = tmp_path / "model.txt"
    model_path.write_text("", encoding="utf-8")
    backend = build_asr_backend(_settings(tmp_path, model_path=model_path))

    with pytest.raises(RuntimeError, match="empty transcript"):
        backend.transcribe(_request())


def test_external_worker_pipeline_parses_segments_json_v1(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text(
        """
{
  "result_format": "segments_json_v1",
  "segments": [
    {"start_sample": 0, "end_sample": 800, "text": "こんにちは"},
    {
      "start_sample": 800,
      "end_sample": 1600,
      "text": "世界",
      "speaker_label": "speaker-1"
    }
  ]
}
""",
        encoding="utf-8",
    )
    backend = build_asr_backend(
        _settings(
            tmp_path,
            model_path=model_path,
            result_format="segments_json_v1",
        )
    )

    transcript = backend.transcribe(_request())

    assert len(transcript.segments) == 2
    assert transcript.segments[0].start_sample == 0
    assert transcript.segments[0].end_sample == 800
    assert transcript.segments[0].text == "こんにちは"
    assert transcript.segments[0].speaker_label is None
    assert transcript.segments[1].start_sample == 800
    assert transcript.segments[1].end_sample == 1600
    assert transcript.segments[1].text == "世界"
    assert transcript.segments[1].speaker_label == "speaker-1"
    assert asr_transcript_text(transcript) == "こんにちは 世界"


@pytest.mark.parametrize(
    ("output_text", "message"),
    [
        ("not json", "malformed segments_json_v1 JSON"),
        ('{"result_format":"segments_json_v1","segments":[]}', "segments must not be empty"),
        (
            '{"result_format":"segments_json_v1","segments":[{"start_sample":0,"end_sample":1601,"text":"x"}]}',
            "end_sample exceeds request sample count",
        ),
        (
            '{"result_format":"segments_json_v1","segments":[{"start_sample":1,"end_sample":2,"text":"x"},{"start_sample":0,"end_sample":1,"text":"y"}]}',
            "segments must be sorted and non-overlapping",
        ),
        (
            '{"result_format":"segments_json_v1","segments":[{"start_sample":0,"end_sample":1,"text":"x","speaker_label":""}]}',
            "speaker_label must be a non-empty string",
        ),
    ],
)
def test_external_worker_pipeline_rejects_malformed_segments_json_v1(
    tmp_path: Path,
    output_text: str,
    message: str,
) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text(output_text, encoding="utf-8")
    backend = build_asr_backend(
        _settings(
            tmp_path,
            model_path=model_path,
            result_format="segments_json_v1",
        )
    )

    with pytest.raises(RuntimeError, match=message):
        backend.transcribe(_request())
