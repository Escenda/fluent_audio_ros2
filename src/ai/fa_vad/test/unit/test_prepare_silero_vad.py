from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


PACKAGE_ROOT = Path(__file__).parents[2]
PREPARER = PACKAGE_ROOT / "scripts" / "prepare_silero_vad"


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


def _write_silero_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "hubconf.py").write_text("def silero_vad():\n    return None\n", encoding="utf-8")
    return path


def _copy_prepare_package(tmp_path: Path) -> PreparePackage:
    root_dir = tmp_path / "fluent_audio_ros2"
    package_dir = root_dir / "src" / "ai" / "fa_vad"
    script_dir = package_dir / "scripts"
    script_dir.mkdir(parents=True)
    (root_dir / "PRODUCT_OWNER_ROLE.md").write_text("test root marker\n", encoding="utf-8")
    (package_dir / "package.xml").write_text("<package></package>\n", encoding="utf-8")
    preparer = script_dir / PREPARER.name
    shutil.copy2(PREPARER, preparer)
    preparer.chmod(0o755)
    worker = _write_executable(script_dir / "silero_vad_worker", "#!/usr/bin/env bash\nexit 0\n")
    return PreparePackage(
        root_dir=root_dir,
        package_dir=package_dir,
        script_dir=script_dir,
        preparer=preparer,
        worker=worker,
    )


def _run_prepare(
    script: Path,
    args: tuple[str, ...],
    *,
    env_updates: dict[str, str] | None = None,
    env_removals: tuple[str, ...] = ("FLUENT_AUDIO_VAD_PREPARE_TRACE_FILE",),
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
                '"$FLUENT_AUDIO_VAD_MODEL_DIR" '
                '"$FLUENT_AUDIO_VAD_PROVIDER" '
                '"$FLUENT_AUDIO_VAD_WORKER" '
                '"$FLUENT_AUDIO_VAD_PREPARE_TRACE_FILE"'
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
    assert len(values) == 4
    return values


def _write_fake_git(path: Path) -> None:
    _write_executable(
        path,
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" != "clone" ]]; then
  printf 'unexpected git command: %s\\n' "$*" >&2
  exit 12
fi
DESTINATION="${@: -1}"
mkdir -p "${DESTINATION}"
printf 'def silero_vad():\\n    return None\\n' > "${DESTINATION}/hubconf.py"
""",
    )


def test_prepare_outputs_sourceable_host_and_vlabor_env_blocks(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    model_dir = _write_silero_repo(package.package_dir / "models" / "silero-vad")

    completed = _run_prepare(package.preparer, ("--model-dir", str(model_dir)))

    assert completed.returncode == 0
    assert "Trace file:" in completed.stderr
    env_file = tmp_path / "vad.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    host_model_dir, host_provider, host_worker_path, trace_file = _source_env_block(env_file)
    assert (host_model_dir, host_provider, host_worker_path) == (
        str(model_dir.resolve(strict=True)),
        "cpu",
        str(package.worker.resolve(strict=True)),
    )
    assert Path(trace_file).is_file()
    assert Path(trace_file).parent == package.package_dir / "traces" / "prepare_silero_vad"
    trace_text = Path(trace_file).read_text(encoding="utf-8")
    assert "started_at=" in trace_text
    assert "ended_at=" in trace_text
    assert "status=success" in trace_text
    assert f"model_dir={model_dir.resolve(strict=True)}" in trace_text
    assert "provider=cpu" in trace_text
    assert f"worker_path={package.worker.resolve(strict=True)}" in trace_text
    assert "stage_start=" in trace_text
    assert "validate_silero_repo" in trace_text
    assert "stage_finish=" in trace_text

    vlabor_values = _source_env_block(env_file, target="vlabor")
    assert vlabor_values[:3] == (
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_vad/models/silero-vad",
        "cpu",
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_vad/scripts/silero_vad_worker",
    )
    expected_container_trace = (
        "/ros2_ws/src/fluent_audio_ros2/"
        f"{Path(trace_file).relative_to(package.root_dir).as_posix()}"
    )
    assert vlabor_values[3] == expected_container_trace


def test_prepare_fails_closed_when_existing_model_dir_lacks_hubconf(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    model_dir = package.package_dir / "models" / "not-silero"
    model_dir.mkdir(parents=True)
    trace_file = tmp_path / "invalid-repo-trace.log"

    completed = _run_prepare(
        package.preparer,
        ("--trace-file", str(trace_file), "--model-dir", str(model_dir)),
    )

    reason = (
        "Silero VAD model directory must be a local torch.hub repository with hubconf.py: "
        f"{model_dir.resolve(strict=True)}"
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert f"ERROR: {reason}" in completed.stderr
    trace_text = trace_file.read_text(encoding="utf-8")
    assert "status=failure" in trace_text
    assert f"failure_reason={reason}" in trace_text
    assert "failure_reason=command exited with status 1" not in trace_text


def test_prepare_clones_missing_package_local_repo_with_fake_git(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_git(bin_dir / "git")
    model_dir = package.package_dir / "models" / "silero-vad"

    completed = _run_prepare(
        package.preparer,
        ("--model-dir", str(model_dir), "--repo-url", "https://example.invalid/silero.git"),
        env_updates={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert completed.returncode == 0
    assert "Cloning Silero VAD" in completed.stderr
    assert (model_dir / "hubconf.py").is_file()
    env_file = tmp_path / "cloned-vad.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    assert _source_env_block(env_file)[:3] == (
        str(model_dir.resolve(strict=True)),
        "cpu",
        str(package.worker.resolve(strict=True)),
    )
