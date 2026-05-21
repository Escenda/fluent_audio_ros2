from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


PACKAGE_ROOT = Path(__file__).parents[2]
PREPARER = PACKAGE_ROOT / "scripts" / "prepare_nemo_rnnt_streaming_asr"
WORKER = PACKAGE_ROOT / "scripts" / "nemo_rnnt_streaming_worker"


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
    package_dir = root_dir / "src" / "ai" / "fa_asr"
    script_dir = package_dir / "scripts"
    script_dir.mkdir(parents=True)
    (root_dir / "PRODUCT_OWNER_ROLE.md").write_text("test root marker\n", encoding="utf-8")
    preparer = script_dir / PREPARER.name
    shutil.copy2(PREPARER, preparer)
    preparer.chmod(0o755)
    worker = script_dir / WORKER.name
    shutil.copy2(WORKER, worker)
    worker.chmod(0o755)
    return PreparePackage(
        root_dir=root_dir,
        package_dir=package_dir,
        script_dir=script_dir,
        preparer=preparer,
        worker=worker,
    )


def _write_fake_curl(path: Path) -> None:
    _write_executable(
        path,
        """#!/usr/bin/env bash
set -euo pipefail
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${OUTPUT}" ]]; then
  printf 'missing output path\\n' >&2
  exit 21
fi
printf 'fake nemo model\\n' > "${OUTPUT}"
""",
    )


def _run_prepare(
    script: Path,
    args: tuple[str, ...],
    *,
    env_updates: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
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


def _source_env_block(env_file: Path, *, target: str | None = None) -> tuple[str, str, str, str]:
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
                'printf "%s\\n%s\\n%s\\n%s\\n" '
                '"$FLUENT_AUDIO_ASR_BACKEND" '
                '"$FLUENT_AUDIO_ASR_MODEL_PATH" '
                '"$FLUENT_AUDIO_ASR_WORKER" '
                '"$FLUENT_AUDIO_ASR_PREPARE_TRACE_FILE"'
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
    return tuple(completed.stdout.splitlines())


def test_prepare_downloads_ignored_model_and_outputs_sourceable_env(
    tmp_path: Path,
) -> None:
    package = _copy_prepare_package(tmp_path)
    bin_dir = tmp_path / "bin"
    _write_fake_curl(bin_dir / "curl")
    model_dir = package.package_dir / "models" / "nemo_rnnt_streaming"
    trace_file = tmp_path / "trace.log"

    completed = _run_prepare(
        package.preparer,
        (
            "--models-dir",
            str(model_dir),
            "--trace-file",
            str(trace_file),
        ),
        env_updates={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert completed.returncode == 0
    assert "Trace file:" in completed.stderr
    env_file = tmp_path / "asr.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    model_path = model_dir / "nemotron-speech-streaming-en-0.6b.nemo"
    backend, host_model_path, host_worker_path, host_trace_file = _source_env_block(env_file)
    assert backend == "nemo_rnnt_streaming"
    assert host_model_path == str(model_path.resolve(strict=True))
    assert host_worker_path == str(package.worker.resolve(strict=True))
    assert host_trace_file == str(trace_file)
    trace_text = trace_file.read_text(encoding="utf-8")
    assert "status=success" in trace_text
    assert "model_id=nemotron-speech-streaming-en-0.6b" in trace_text
    assert f"model_path={model_path.resolve(strict=True)}" in trace_text

    vlabor_backend, vlabor_model, vlabor_worker, vlabor_trace = _source_env_block(
        env_file,
        target="vlabor",
    )
    assert vlabor_backend == "nemo_rnnt_streaming"
    assert vlabor_model == (
        "/ros2_ws/src/fluent_audio_ros2/"
        f"{model_path.relative_to(package.root_dir).as_posix()}"
    )
    assert vlabor_worker == (
        "/ros2_ws/src/fluent_audio_ros2/"
        f"{package.worker.relative_to(package.root_dir).as_posix()}"
    )
    assert vlabor_trace == str(trace_file)


def test_prepare_fails_closed_on_unknown_model_id(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)

    completed = _run_prepare(package.preparer, ("--model-id", "unknown-model"))

    assert completed.returncode == 1
    assert "unsupported model id" in completed.stderr


def test_prepare_fails_closed_when_download_fails(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    bin_dir = tmp_path / "bin"
    _write_executable(
        bin_dir / "curl",
        "#!/usr/bin/env bash\nprintf 'download failed\\n' >&2\nexit 22\n",
    )

    completed = _run_prepare(
        package.preparer,
        ("--models-dir", str(package.package_dir / "models" / "nemo_rnnt_streaming")),
        env_updates={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert completed.returncode == 1
    assert "failed to download model" in completed.stderr
