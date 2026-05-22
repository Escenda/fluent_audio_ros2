from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


PACKAGE_ROOT = Path(__file__).parents[2]
PREPARER = PACKAGE_ROOT / "scripts" / "prepare_nemo_offline_transcribe_asr"
WORKER = PACKAGE_ROOT / "scripts" / "nemo_offline_transcribe_worker"
MODEL_ID = "parakeet-1.1b-multilingual-offline"
ARTIFACT = "nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal:trainable_v1.0"
MODEL_FILE = "Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo"


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


def _write_fake_ngc(path: Path, *, produce_model: bool = True) -> None:
    model_file = MODEL_FILE if produce_model else "unexpected-model.nemo"
    _write_executable(
        path,
        f"""#!/usr/bin/env bash
set -euo pipefail
DEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${{DEST}}" ]]; then
  printf 'missing --dest\\n' >&2
  exit 21
fi
mkdir -p "${{DEST}}/artifact"
printf 'fake nemo model\\n' > "${{DEST}}/artifact/{model_file}"
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
                'printf "%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n" '
                '"$FLUENT_AUDIO_ASR_BACKEND" '
                '"$FLUENT_AUDIO_NEMO_OFFLINE_TRANSCRIBE_MODEL_PATH" '
                '"$FLUENT_AUDIO_NEMO_OFFLINE_TRANSCRIBE_WORKER" '
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


def test_prepare_downloads_ngc_artifact_and_outputs_sourceable_env(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    bin_dir = tmp_path / "bin"
    _write_fake_ngc(bin_dir / "ngc")
    model_dir = package.package_dir / "models" / "nemo_offline_transcribe"
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

    assert completed.returncode == 0, completed.stderr
    assert "Trace file:" in completed.stderr
    env_file = tmp_path / "asr.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    model_path = model_dir / MODEL_FILE
    (
        backend,
        explicit_model_path,
        explicit_worker_path,
        generic_model_path,
        generic_worker_path,
        host_trace_file,
    ) = _source_env_block(env_file)
    assert backend == "nemo_offline_transcribe"
    assert explicit_model_path == str(model_path.resolve(strict=True))
    assert explicit_worker_path == str(package.worker.resolve(strict=True))
    assert generic_model_path == explicit_model_path
    assert generic_worker_path == explicit_worker_path
    assert host_trace_file == str(trace_file)
    trace_text = trace_file.read_text(encoding="utf-8")
    assert "status=success" in trace_text
    assert f"model_id={MODEL_ID}" in trace_text
    assert f"artifact={ARTIFACT}" in trace_text
    assert f"model_path={model_path.resolve(strict=True)}" in trace_text
    assert f"worker_path={package.worker.resolve(strict=True)}" in trace_text

    (
        vlabor_backend,
        vlabor_model,
        vlabor_worker,
        vlabor_generic_model,
        vlabor_generic_worker,
        vlabor_trace,
    ) = _source_env_block(env_file, target="vlabor")
    assert vlabor_backend == "nemo_offline_transcribe"
    assert vlabor_model == (
        "/ros2_ws/src/fluent_audio_ros2/"
        f"{model_path.relative_to(package.root_dir).as_posix()}"
    )
    assert vlabor_worker == (
        "/ros2_ws/src/fluent_audio_ros2/"
        f"{package.worker.relative_to(package.root_dir).as_posix()}"
    )
    assert vlabor_generic_model == vlabor_model
    assert vlabor_generic_worker == vlabor_worker
    assert vlabor_trace == str(trace_file)


def test_prepare_rejects_unsupported_model_id(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    trace_file = tmp_path / "trace.log"

    completed = _run_prepare(
        package.preparer,
        ("--model-id", "unknown-model", "--trace-file", str(trace_file)),
    )

    assert completed.returncode == 1
    assert "unsupported model id" in completed.stderr
    assert "failure_reason=unsupported model id: unknown-model" in trace_file.read_text(
        encoding="utf-8",
    )


def test_prepare_rejects_unsupported_artifact(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)

    completed = _run_prepare(
        package.preparer,
        ("--artifact", "nvidia/riva/unknown:1"),
    )

    assert completed.returncode == 1
    assert "unsupported artifact" in completed.stderr


def test_prepare_fails_when_ngc_is_missing_and_model_does_not_exist(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    trace_file = tmp_path / "trace.log"

    completed = _run_prepare(
        package.preparer,
        (
            "--models-dir",
            str(package.package_dir / "models" / "nemo_offline_transcribe"),
            "--trace-file",
            str(trace_file),
        ),
        env_updates={"PATH": "/usr/bin:/bin"},
    )

    assert completed.returncode == 1
    assert "ngc is required" in completed.stderr
    assert "failure_reason=ngc is required" in trace_file.read_text(encoding="utf-8")


def test_prepare_fails_when_ngc_does_not_produce_expected_nemo(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    bin_dir = tmp_path / "bin"
    _write_fake_ngc(bin_dir / "ngc", produce_model=False)

    completed = _run_prepare(
        package.preparer,
        (
            "--models-dir",
            str(package.package_dir / "models" / "nemo_offline_transcribe"),
        ),
        env_updates={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert completed.returncode == 1
    assert "did not produce expected .nemo file" in completed.stderr


def test_prepare_uses_existing_model_without_ngc(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    model_dir = package.package_dir / "models" / "nemo_offline_transcribe"
    model_dir.mkdir(parents=True)
    model_path = model_dir / MODEL_FILE
    model_path.write_text("existing nemo model\n", encoding="utf-8")

    completed = _run_prepare(
        package.preparer,
        ("--models-dir", str(model_dir)),
        env_updates={"PATH": "/usr/bin:/bin"},
    )

    assert completed.returncode == 0, completed.stderr
    env_file = tmp_path / "asr.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    _, explicit_model_path, *_ = _source_env_block(env_file)
    assert explicit_model_path == str(model_path.resolve(strict=True))
