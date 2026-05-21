import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest


PACKAGE_ROOT = Path(__file__).parents[2]
WORKER = PACKAGE_ROOT / "scripts" / "nemo_offline_transcribe_worker"


def _write_fake_nemo_package(tmp_path: Path) -> Path:
    package_root = tmp_path / "fake_python"
    models_path = package_root / "nemo" / "collections" / "asr"
    models_path.mkdir(parents=True)
    for init_path in (
        package_root / "nemo" / "__init__.py",
        package_root / "nemo" / "collections" / "__init__.py",
        models_path / "__init__.py",
    ):
        init_path.write_text("", encoding="utf-8")
    (models_path / "models.py").write_text(
        """from pathlib import Path
import json
import os
import struct
import sys
import wave


class ASRModel:
    @staticmethod
    def restore_from(path):
        if os.environ.get("FAKE_NEMO_RESTORE_FAIL") == "1":
            raise RuntimeError("forced restore failure")
        return FakeModel(path)


class FakeModel:
    def __init__(self, path):
        self.path = path
        self.cfg = {
            "preprocessor": {"sample_rate": int(os.environ.get("FAKE_NEMO_SAMPLE_RATE", "16000"))},
            "languages": ["ja", "en"],
        }
        if os.environ.get("FAKE_NEMO_NO_TRANSCRIBE") == "1":
            delattr(self.__class__, "transcribe")

    def eval(self):
        print("fake eval log")

    def transcribe(self, audio, batch_size=1, return_hypotheses=False, verbose=False, language=None):
        print("fake transcribe log")
        wav_path = Path(audio[0])
        with wave.open(str(wav_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_bytes = wav_file.readframes(frame_count)
            record = {
                "argv_audio_count": len(audio),
                "batch_size": batch_size,
                "return_hypotheses": return_hypotheses,
                "verbose": verbose,
                "language": language,
                "sample_rate": wav_file.getframerate(),
                "channels": wav_file.getnchannels(),
                "sample_width": wav_file.getsampwidth(),
                "frame_count": frame_count,
                "samples": [sample for (sample,) in struct.iter_unpack("<h", frame_bytes)],
            }
        Path(os.environ["FAKE_NEMO_RECORD"]).write_text(
            json.dumps(record, ensure_ascii=False),
            encoding="utf-8",
        )
        transcript = os.environ.get("FAKE_NEMO_TRANSCRIPT", "認識しました")
        return [transcript]
""",
        encoding="utf-8",
    )
    return package_root


def _write_float32le(path: Path, samples: tuple[float, ...]) -> Path:
    path.write_bytes(b"".join(struct.pack("<f", sample) for sample in samples))
    return path


def _write_model(path: Path) -> Path:
    path.write_bytes(b"fake nemo")
    return path


def _run_worker(
    tmp_path: Path,
    args: tuple[str, ...],
    *,
    env_updates: dict[str, str] | None = None,
    fake_nemo_package: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if fake_nemo_package:
        env["PYTHONPATH"] = str(_write_fake_nemo_package(tmp_path))
    else:
        env["PYTHONPATH"] = str(tmp_path)
    env["FAKE_NEMO_RECORD"] = str(tmp_path / "record.json")
    if env_updates is not None:
        env.update(env_updates)
    return subprocess.run(
        (sys.executable, "-S", str(WORKER)) + args,
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
        check=False,
    )


def _read_record(path: Path) -> dict[str, str | int | bool | list[int]]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_health_loads_model_and_keeps_stdout_to_protocol_only(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "parakeet.nemo")

    completed = _run_worker(
        tmp_path,
        (
            "health",
            "--model",
            str(model_path),
            "--language",
            "ja",
            "--sample-rate",
            "16000",
            "--channels",
            "1",
        ),
    )

    assert completed.returncode == 0
    assert completed.stdout == "ok\n"
    assert "fake eval log" in completed.stderr


def test_health_fails_when_nemo_package_is_missing(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "parakeet.nemo")

    completed = _run_worker(
        tmp_path,
        (
            "health",
            "--model",
            str(model_path),
            "--language",
            "ja",
            "--sample-rate",
            "16000",
            "--channels",
            "1",
        ),
        fake_nemo_package=False,
    )

    assert completed.returncode != 0
    assert "NeMo ASR module is unavailable" in completed.stderr
    assert completed.stdout == ""


def test_health_fails_when_model_has_no_offline_transcribe(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "parakeet.nemo")

    completed = _run_worker(
        tmp_path,
        (
            "health",
            "--model",
            str(model_path),
            "--language",
            "ja",
            "--sample-rate",
            "16000",
            "--channels",
            "1",
        ),
        env_updates={"FAKE_NEMO_NO_TRANSCRIBE": "1"},
    )

    assert completed.returncode != 0
    assert "offline transcribe" in completed.stderr


def test_transcribe_converts_raw_float32le_to_explicit_wav_bridge(
    tmp_path: Path,
) -> None:
    model_path = _write_model(tmp_path / "parakeet.nemo")
    audio_path = _write_float32le(tmp_path / "input.f32", (-1.0, 0.0, 1.0))

    completed = _run_worker(
        tmp_path,
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--channels",
            "1",
            "--language",
            "ja",
            "--result-format",
            "plain_text",
        ),
    )

    assert completed.returncode == 0
    assert completed.stdout == "認識しました\n"
    assert "fake transcribe log" in completed.stderr
    record = _read_record(tmp_path / "record.json")
    assert record["language"] == "ja"
    assert record["sample_rate"] == 16000
    assert record["channels"] == 1
    assert record["sample_width"] == 2
    assert record["frame_count"] == 3
    assert record["samples"] == [-32768, 0, 32767]


def test_transcribe_emits_strict_segments_json_v1(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "parakeet.nemo")
    audio_path = _write_float32le(tmp_path / "input.f32", (0.25, 0.5))

    completed = _run_worker(
        tmp_path,
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--channels",
            "1",
            "--language",
            "ja",
            "--result-format",
            "segments_json_v1",
        ),
    )

    assert completed.returncode == 0
    document = json.loads(completed.stdout)
    assert set(document.keys()) == {"result_format", "segments"}
    assert document["result_format"] == "segments_json_v1"
    assert document["segments"] == [
        {"start_sample": 0, "end_sample": 2, "text": "認識しました"}
    ]


def test_transcribe_fails_empty_transcript(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "parakeet.nemo")
    audio_path = _write_float32le(tmp_path / "input.f32", (0.25,))

    completed = _run_worker(
        tmp_path,
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--channels",
            "1",
            "--language",
            "ja",
            "--result-format",
            "plain_text",
        ),
        env_updates={"FAKE_NEMO_TRANSCRIPT": ""},
    )

    assert completed.returncode != 0
    assert "empty transcript" in completed.stderr
    assert completed.stdout == ""


@pytest.mark.parametrize(
    ("samples", "expected"),
    (
        ((float("nan"),), "not finite"),
        ((1.25,), "outside normalized"),
    ),
)
def test_transcribe_rejects_invalid_float32le_samples(
    tmp_path: Path,
    samples: tuple[float, ...],
    expected: str,
) -> None:
    model_path = _write_model(tmp_path / "parakeet.nemo")
    audio_path = _write_float32le(tmp_path / "input.f32", samples)

    completed = _run_worker(
        tmp_path,
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--channels",
            "1",
            "--language",
            "ja",
            "--result-format",
            "plain_text",
        ),
    )

    assert completed.returncode != 0
    assert expected in completed.stderr
    assert completed.stdout == ""
