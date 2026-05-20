import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess

import pytest


PACKAGE_ROOT = Path(__file__).parents[2]
WORKER = PACKAGE_ROOT / "scripts" / "whisper_cpp_worker"


@dataclass(frozen=True)
class FakeWhisperRecord:
    argv: tuple[str, ...]
    ld_library_path: str
    language: str
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int
    samples: tuple[int, ...]


def _write_float32le(path: Path, samples: tuple[float, ...]) -> Path:
    path.write_bytes(b"".join(struct.pack("<f", sample) for sample in samples))
    return path


def _write_fake_model(path: Path) -> Path:
    path.write_bytes(b"fake ggml model")
    return path


def _write_fake_whisper_cli(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import struct
import sys
import wave


def _flag_value(flag):
    for index, value in enumerate(sys.argv):
        if value == flag and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    raise RuntimeError(f"missing flag: {flag}")


if "--help" in sys.argv:
    print("fake whisper-cli")
    raise SystemExit(0)
if os.environ.get("FA_ASR_FAKE_WHISPER_FAIL") == "1":
    print("forced whisper failure", file=sys.stderr)
    raise SystemExit(7)

model_path = Path(_flag_value("-m"))
wav_path = Path(_flag_value("-f"))
language = _flag_value("-l")
output_base = Path(_flag_value("-of"))
if "-otxt" not in sys.argv:
    raise RuntimeError("text output flag is required")
if not model_path.is_file():
    raise RuntimeError("model is required")

with wave.open(str(wav_path), "rb") as wav_file:
    frame_count = wav_file.getnframes()
    frame_bytes = wav_file.readframes(frame_count)
    samples = [sample for (sample,) in struct.iter_unpack("<h", frame_bytes)]
    record = {
        "argv": sys.argv[1:],
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "language": language,
        "sample_rate": wav_file.getframerate(),
        "channels": wav_file.getnchannels(),
        "sample_width": wav_file.getsampwidth(),
        "frame_count": frame_count,
        "samples": samples,
    }

Path(os.environ["FA_ASR_FAKE_WHISPER_RECORD"]).write_text(
    json.dumps(record, ensure_ascii=False),
    encoding="utf-8",
)
Path(str(output_base) + ".txt").write_text(
    os.environ.get("FA_ASR_FAKE_WHISPER_TRANSCRIPT", "fake transcript") + "\\n",
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run_worker(
    args: tuple[str, ...],
    *,
    record_path: Path,
    transcript: str = "認識しました",
    fail_cli: bool = False,
    env_updates: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["FA_ASR_FAKE_WHISPER_RECORD"] = str(record_path)
    env["FA_ASR_FAKE_WHISPER_TRANSCRIPT"] = transcript
    if fail_cli:
        env["FA_ASR_FAKE_WHISPER_FAIL"] = "1"
    if env_updates is not None:
        env.update(env_updates)
    return subprocess.run(
        (str(WORKER),) + args,
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
        check=False,
    )


def _read_record(path: Path) -> FakeWhisperRecord:
    document = json.loads(path.read_text(encoding="utf-8"))
    return FakeWhisperRecord(
        argv=tuple(document["argv"]),
        ld_library_path=document["ld_library_path"],
        language=document["language"],
        sample_rate=document["sample_rate"],
        channels=document["channels"],
        sample_width=document["sample_width"],
        frame_count=document["frame_count"],
        samples=tuple(document["samples"]),
    )


def test_health_accepts_existing_model_and_executable_binary(tmp_path: Path) -> None:
    model_path = _write_fake_model(tmp_path / "ggml-base.en.bin")
    binary_path = _write_fake_whisper_cli(tmp_path / "whisper-cli")
    record_path = tmp_path / "record.json"

    completed = _run_worker(
        (
            "health",
            "--model",
            str(model_path),
            "--language",
            "ja",
            "--binary",
            str(binary_path),
        ),
        record_path=record_path,
    )

    assert completed.returncode == 0
    assert completed.stdout == "ok\n"
    assert completed.stderr == ""
    assert not record_path.exists()


def test_transcribe_converts_raw_float32le_mono_to_expected_wav(
    tmp_path: Path,
) -> None:
    model_path = _write_fake_model(tmp_path / "ggml-base.en.bin")
    binary_path = _write_fake_whisper_cli(tmp_path / "whisper-cli")
    audio_path = _write_float32le(tmp_path / "input.f32", (-1.0, -0.5, 0.0, 0.5, 1.0))
    record_path = tmp_path / "record.json"

    completed = _run_worker(
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--language",
            "ja",
            "--binary",
            str(binary_path),
        ),
        record_path=record_path,
    )

    assert completed.returncode == 0
    assert completed.stdout == "認識しました\n"
    assert completed.stderr == ""
    record = _read_record(record_path)
    assert record.language == "ja"
    assert record.sample_rate == 16000
    assert record.channels == 1
    assert record.sample_width == 2
    assert record.frame_count == 5
    assert record.samples == (-32768, -16384, 0, 16384, 32767)
    assert record.argv[0:2] == ("-m", str(model_path.resolve(strict=True)))
    assert "-otxt" in record.argv


def test_transcribe_prepends_package_local_whisper_cpp_library_dirs(
    tmp_path: Path,
) -> None:
    model_path = _write_fake_model(tmp_path / "ggml-base.en.bin")
    build_dir = tmp_path / "fa_asr" / "tools" / "whisper.cpp" / "build"
    build_src_dir = build_dir / "src"
    build_ggml_src_dir = build_dir / "ggml" / "src"
    build_src_dir.mkdir(parents=True)
    build_ggml_src_dir.mkdir(parents=True)
    binary_path = _write_fake_whisper_cli(build_dir / "bin" / "whisper-cli")
    audio_path = _write_float32le(tmp_path / "input.f32", (0.25,))
    record_path = tmp_path / "record.json"
    existing_library_path = str(tmp_path / "existing_lib")

    completed = _run_worker(
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--language",
            "ja",
            "--binary",
            str(binary_path),
        ),
        record_path=record_path,
        env_updates={"LD_LIBRARY_PATH": existing_library_path},
    )

    assert completed.returncode == 0
    record = _read_record(record_path)
    assert record.ld_library_path == os.pathsep.join(
        (
            str(build_src_dir.resolve(strict=True)),
            str(build_ggml_src_dir.resolve(strict=True)),
            existing_library_path,
        )
    )


def test_transcribe_writes_explicit_output_file_when_requested(tmp_path: Path) -> None:
    model_path = _write_fake_model(tmp_path / "ggml-base.en.bin")
    binary_path = _write_fake_whisper_cli(tmp_path / "whisper-cli")
    audio_path = _write_float32le(tmp_path / "input.f32", (0.125, 0.25))
    record_path = tmp_path / "record.json"
    output_path = tmp_path / "result" / "transcript.txt"

    completed = _run_worker(
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--language",
            "en",
            "--output",
            str(output_path),
            "--binary",
            str(binary_path),
        ),
        record_path=record_path,
        transcript="hello world",
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert output_path.read_text(encoding="utf-8") == "hello world\n"
    assert _read_record(record_path).samples == (4096, 8192)


def test_transcribe_rejects_missing_model_before_calling_whisper_cli(tmp_path: Path) -> None:
    binary_path = _write_fake_whisper_cli(tmp_path / "whisper-cli")
    audio_path = _write_float32le(tmp_path / "input.f32", (0.0,))
    record_path = tmp_path / "record.json"

    completed = _run_worker(
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(tmp_path / "missing.bin"),
            "--sample-rate",
            "16000",
            "--language",
            "ja",
            "--binary",
            str(binary_path),
        ),
        record_path=record_path,
    )

    assert completed.returncode == 1
    assert "model file does not exist" in completed.stderr
    assert not record_path.exists()


def test_transcribe_rejects_missing_whisper_cli_binary(tmp_path: Path) -> None:
    model_path = _write_fake_model(tmp_path / "ggml-base.en.bin")
    audio_path = _write_float32le(tmp_path / "input.f32", (0.0,))
    record_path = tmp_path / "record.json"

    completed = _run_worker(
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--language",
            "ja",
            "--binary",
            str(tmp_path / "missing-whisper-cli"),
        ),
        record_path=record_path,
    )

    assert completed.returncode == 1
    assert "whisper-cli binary does not exist" in completed.stderr
    assert not record_path.exists()


def test_transcribe_rejects_non_16000_sample_rate_without_resampling(tmp_path: Path) -> None:
    model_path = _write_fake_model(tmp_path / "ggml-base.en.bin")
    binary_path = _write_fake_whisper_cli(tmp_path / "whisper-cli")
    audio_path = _write_float32le(tmp_path / "input.f32", (0.0,))
    record_path = tmp_path / "record.json"

    completed = _run_worker(
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "48000",
            "--language",
            "ja",
            "--binary",
            str(binary_path),
        ),
        record_path=record_path,
    )

    assert completed.returncode == 1
    assert "Resampling must happen before fa_asr" in completed.stderr
    assert not record_path.exists()


@pytest.mark.parametrize(
    ("samples", "expected_error"),
    [
        ((math.inf,), "not finite"),
        ((1.01,), "outside normalized"),
        ((-1.01,), "outside normalized"),
    ],
)
def test_transcribe_rejects_non_finite_or_non_normalized_float32le_audio(
    tmp_path: Path,
    samples: tuple[float, ...],
    expected_error: str,
) -> None:
    model_path = _write_fake_model(tmp_path / "ggml-base.en.bin")
    binary_path = _write_fake_whisper_cli(tmp_path / "whisper-cli")
    audio_path = _write_float32le(tmp_path / "input.f32", samples)
    record_path = tmp_path / "record.json"

    completed = _run_worker(
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--language",
            "ja",
            "--binary",
            str(binary_path),
        ),
        record_path=record_path,
    )

    assert completed.returncode == 1
    assert expected_error in completed.stderr
    assert not record_path.exists()


def test_transcribe_rejects_non_float32le_byte_payload(tmp_path: Path) -> None:
    model_path = _write_fake_model(tmp_path / "ggml-base.en.bin")
    binary_path = _write_fake_whisper_cli(tmp_path / "whisper-cli")
    audio_path = tmp_path / "input.f32"
    audio_path.write_bytes(b"\x00\x01")
    record_path = tmp_path / "record.json"

    completed = _run_worker(
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--language",
            "ja",
            "--binary",
            str(binary_path),
        ),
        record_path=record_path,
    )

    assert completed.returncode == 1
    assert "raw float32le mono" in completed.stderr
    assert not record_path.exists()


def test_transcribe_fails_closed_when_whisper_cli_fails(tmp_path: Path) -> None:
    model_path = _write_fake_model(tmp_path / "ggml-base.en.bin")
    binary_path = _write_fake_whisper_cli(tmp_path / "whisper-cli")
    audio_path = _write_float32le(tmp_path / "input.f32", (0.0,))
    record_path = tmp_path / "record.json"

    completed = _run_worker(
        (
            "transcribe",
            "--audio",
            str(audio_path),
            "--model",
            str(model_path),
            "--sample-rate",
            "16000",
            "--language",
            "ja",
            "--binary",
            str(binary_path),
        ),
        record_path=record_path,
        fail_cli=True,
    )

    assert completed.returncode == 1
    assert "whisper-cli transcribe failed" in completed.stderr
    assert "forced whisper failure" in completed.stderr
    assert not record_path.exists()
