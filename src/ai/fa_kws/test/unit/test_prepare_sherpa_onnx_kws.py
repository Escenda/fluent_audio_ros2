from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


PACKAGE_ROOT = Path(__file__).parents[2]
PREPARER = PACKAGE_ROOT / "scripts" / "prepare_sherpa_onnx_kws"
MODEL_ID = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
ORIGINAL_ENCODER = "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
ORIGINAL_DECODER = "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
ORIGINAL_JOINER = "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
ORIGINAL_TOKENS = "tokens.txt"
KEYWORDS_TEXT = "▁HE Y ▁AS P A\n"


@dataclass(frozen=True)
class PreparePackage:
    root_dir: Path
    package_dir: Path
    script_dir: Path
    preparer: Path
    worker: Path


def _write_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _copy_prepare_package(tmp_path: Path) -> PreparePackage:
    root_dir = tmp_path / "fluent_audio_ros2"
    package_dir = root_dir / "src" / "ai" / "fa_kws"
    script_dir = package_dir / "scripts"
    script_dir.mkdir(parents=True)
    (root_dir / "PRODUCT_OWNER_ROLE.md").write_text("test root marker\n", encoding="utf-8")
    (package_dir / "package.xml").write_text("<package></package>\n", encoding="utf-8")
    preparer = script_dir / PREPARER.name
    shutil.copy2(PREPARER, preparer)
    preparer.chmod(0o755)
    worker = _write_executable(script_dir / "sherpa_onnx_kws_worker", "#!/usr/bin/env bash\nexit 0\n")
    return PreparePackage(
        root_dir=root_dir,
        package_dir=package_dir,
        script_dir=script_dir,
        preparer=preparer,
        worker=worker,
    )


def _write_model_archive(tmp_path: Path, *, include_joiner: bool = True) -> Path:
    archive_root = tmp_path / "archive-root"
    model_dir = archive_root / MODEL_ID
    model_dir.mkdir(parents=True)
    (model_dir / ORIGINAL_ENCODER).write_bytes(b"encoder\n")
    (model_dir / ORIGINAL_DECODER).write_bytes(b"decoder\n")
    if include_joiner:
        (model_dir / ORIGINAL_JOINER).write_bytes(b"joiner\n")
    (model_dir / ORIGINAL_TOKENS).write_bytes(b"<blk>\nAS\nPA\n")
    archive_path = tmp_path / "kws-model.tar.bz2"
    completed = subprocess.run(
        ("tar", "-cjf", str(archive_path), "-C", str(archive_root), MODEL_ID),
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return archive_path


def _write_fake_curl(path: Path, archive_path: Path) -> None:
    _write_executable(
        path,
        f"""#!/usr/bin/env bash
set -euo pipefail
output=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -o)
      output="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${{output}}" ]]; then
  printf 'missing -o output\\n' >&2
  exit 11
fi
cp -- {str(archive_path)!r} "${{output}}"
""",
    )


def _run_prepare(
    script: Path,
    args: tuple[str, ...],
    *,
    env_updates: dict[str, str] | None = None,
    env_removals: tuple[str, ...] = ("FLUENT_AUDIO_KWS_PREPARE_TRACE_FILE",),
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in env_removals:
        environment.pop(name, None)
    if env_updates is not None:
        environment.update(env_updates)
    return subprocess.run(
        (str(script),) + args,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10.0,
        check=False,
    )


def _source_env_block(env_file: Path, *, target: str | None = None) -> tuple[str, ...]:
    environment = os.environ.copy()
    if target is None:
        environment.pop("FLUENT_AUDIO_ENV_TARGET", None)
    else:
        environment["FLUENT_AUDIO_ENV_TARGET"] = target
    completed = subprocess.run(
        (
            "bash",
            "-c",
            (
                'source "$1"; '
                'printf "%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n" '
                '"$FLUENT_AUDIO_KWS_PROVIDER" '
                '"$FLUENT_AUDIO_KWS_WORKER" '
                '"$FLUENT_AUDIO_KWS_ENCODER" '
                '"$FLUENT_AUDIO_KWS_DECODER" '
                '"$FLUENT_AUDIO_KWS_JOINER" '
                '"$FLUENT_AUDIO_KWS_TOKENS" '
                '"$FLUENT_AUDIO_KWS_KEYWORDS" '
                '"$FLUENT_AUDIO_KWS_PREPARE_TRACE_FILE"'
            ),
            "bash",
            str(env_file),
        ),
        capture_output=True,
        text=True,
        env=environment,
        timeout=10.0,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    values = tuple(completed.stdout.splitlines())
    assert len(values) == 8
    return values


def test_prepare_downloads_normalizes_and_outputs_sourceable_env_blocks(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    archive_path = _write_model_archive(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_curl(bin_dir / "curl", archive_path)
    model_dir = package.package_dir / "models" / "sherpa-onnx-kws"

    completed = _run_prepare(
        package.preparer,
        (
            "--model-dir",
            str(model_dir),
            "--download-url",
            "https://example.invalid/kws.tar.bz2",
        ),
        env_updates={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert completed.returncode == 0, completed.stderr
    assert "Downloading sherpa-onnx KWS model archive" in completed.stderr
    assert (model_dir / "encoder.onnx").read_bytes() == b"encoder\n"
    assert (model_dir / "decoder.onnx").read_bytes() == b"decoder\n"
    assert (model_dir / "joiner.onnx").read_bytes() == b"joiner\n"
    assert (model_dir / "tokens.txt").read_bytes() == b"<blk>\nAS\nPA\n"
    assert (model_dir / "keywords.txt").read_text(encoding="utf-8") == KEYWORDS_TEXT

    env_file = tmp_path / "kws.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    host_values = _source_env_block(env_file)
    assert host_values == (
        "cpu",
        str(package.worker.resolve(strict=True)),
        str((model_dir / "encoder.onnx").resolve(strict=True)),
        str((model_dir / "decoder.onnx").resolve(strict=True)),
        str((model_dir / "joiner.onnx").resolve(strict=True)),
        str((model_dir / "tokens.txt").resolve(strict=True)),
        str((model_dir / "keywords.txt").resolve(strict=True)),
        host_values[7],
    )
    trace_file = Path(host_values[7])
    assert trace_file.is_file()
    assert trace_file.parent == package.package_dir / "traces" / "prepare_sherpa_onnx_kws"
    trace_text = trace_file.read_text(encoding="utf-8")
    assert "status=success" in trace_text
    assert f"model_dir={model_dir.resolve(strict=True)}" in trace_text
    assert "provider=cpu" in trace_text
    assert f"worker_path={package.worker.resolve(strict=True)}" in trace_text
    assert "stage_start=" in trace_text
    assert "normalize_model" in trace_text
    assert "stage_finish=" in trace_text

    vlabor_values = _source_env_block(env_file, target="vlabor")
    assert vlabor_values[:7] == (
        "cpu",
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_kws/scripts/sherpa_onnx_kws_worker",
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_kws/models/sherpa-onnx-kws/encoder.onnx",
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_kws/models/sherpa-onnx-kws/decoder.onnx",
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_kws/models/sherpa-onnx-kws/joiner.onnx",
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_kws/models/sherpa-onnx-kws/tokens.txt",
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_kws/models/sherpa-onnx-kws/keywords.txt",
    )
    expected_container_trace = (
        "/ros2_ws/src/fluent_audio_ros2/"
        f"{trace_file.relative_to(package.root_dir).as_posix()}"
    )
    assert vlabor_values[7] == expected_container_trace


def test_prepare_reuses_existing_complete_model_and_validates_provider(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    model_dir = package.package_dir / "models" / "sherpa-onnx-kws"
    model_dir.mkdir(parents=True)
    (model_dir / "encoder.onnx").write_bytes(b"encoder\n")
    (model_dir / "decoder.onnx").write_bytes(b"decoder\n")
    (model_dir / "joiner.onnx").write_bytes(b"joiner\n")
    (model_dir / "tokens.txt").write_bytes(b"tokens\n")
    (model_dir / "keywords.txt").write_text(KEYWORDS_TEXT, encoding="utf-8")

    completed = _run_prepare(package.preparer, ("--model-dir", str(model_dir), "--provider", "coreml"))

    assert completed.returncode == 0, completed.stderr
    assert "Downloading sherpa-onnx KWS model archive" not in completed.stderr
    env_file = tmp_path / "existing-kws.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    assert _source_env_block(env_file)[0] == "coreml"


def test_prepare_fails_closed_when_archive_lacks_required_original_file(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    archive_path = _write_model_archive(tmp_path, include_joiner=False)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_curl(bin_dir / "curl", archive_path)
    trace_file = tmp_path / "missing-joiner-trace.log"
    model_dir = package.package_dir / "models" / "sherpa-onnx-kws"

    completed = _run_prepare(
        package.preparer,
        (
            "--trace-file",
            str(trace_file),
            "--model-dir",
            str(model_dir),
            "--download-url",
            "https://example.invalid/kws.tar.bz2",
        ),
        env_updates={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    expected_reason = (
        "original joiner does not exist or is not a file: "
        f"{model_dir.parent / 'unresolved'}"
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "ERROR: original joiner does not exist or is not a file:" in completed.stderr
    assert ORIGINAL_JOINER in completed.stderr
    trace_text = trace_file.read_text(encoding="utf-8")
    assert "status=failure" in trace_text
    assert "failure_reason=original joiner does not exist or is not a file:" in trace_text
    assert ORIGINAL_JOINER in trace_text
    assert "failure_reason=command exited with status 1" not in trace_text
    assert expected_reason not in completed.stderr
