from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


PACKAGE_ROOT = Path(__file__).parents[2]
PREPARER = PACKAGE_ROOT / "scripts" / "prepare_whisper_cpp_asr"


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
    (package_dir / "package.xml").write_text("<package></package>\n", encoding="utf-8")
    preparer = script_dir / PREPARER.name
    shutil.copy2(PREPARER, preparer)
    preparer.chmod(0o755)
    worker = _write_executable(script_dir / "whisper_cpp_worker", "#!/usr/bin/env bash\nexit 0\n")
    _write_fake_downloader(script_dir / "download_whisper_cpp_model")
    return PreparePackage(
        root_dir=root_dir,
        package_dir=package_dir,
        script_dir=script_dir,
        preparer=preparer,
        worker=worker,
    )


def _write_fake_downloader(path: Path) -> Path:
    return _write_executable(
        path,
        """#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PACKAGE_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
MODEL_ID="${1:-small-q5_1}"
MODELS_DIR="${2:-${PACKAGE_DIR}/models/whisper.cpp}"
mkdir -p "${MODELS_DIR}"
MODEL_PATH="${MODELS_DIR}/ggml-${MODEL_ID}.bin"
if [[ ! -f "${MODEL_PATH}" ]]; then
  printf 'fake model for %s\\n' "${MODEL_ID}" > "${MODEL_PATH}"
fi
printf 'FLUENT_AUDIO_ASR_MODEL_PATH=%s/%s\\n' "$(cd -- "$(dirname -- "${MODEL_PATH}")" >/dev/null 2>&1 && pwd -P)" "$(basename -- "${MODEL_PATH}")"
printf 'FLUENT_AUDIO_ASR_WORKER=%s/whisper_cpp_worker\\n' "${SCRIPT_DIR}"
""",
    )


def _run_prepare(
    script: Path,
    args: tuple[str, ...],
    *,
    env_updates: dict[str, str] | None = None,
    env_removals: tuple[str, ...] = ("FLUENT_AUDIO_WHISPER_CPP_BINARY",),
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
                '"$FLUENT_AUDIO_ASR_MODEL_PATH" '
                '"$FLUENT_AUDIO_ASR_WORKER" '
                '"$FLUENT_AUDIO_WHISPER_CPP_BINARY" '
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


def _link_command(bin_dir: Path, name: str) -> None:
    resolved = shutil.which(name)
    assert resolved is not None
    (bin_dir / name).symlink_to(resolved)


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
printf 'cmake_minimum_required(VERSION 3.8)\\n' > "${DESTINATION}/CMakeLists.txt"
""",
    )


def _write_fake_cmake(path: Path) -> None:
    _write_executable(
        path,
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "-S" ]]; then
  exit 0
fi
if [[ "$1" == "--build" ]]; then
  BUILD_DIR="$2"
  mkdir -p "${BUILD_DIR}/bin"
  printf '#!/usr/bin/env bash\\nexit 0\\n' > "${BUILD_DIR}/bin/whisper-cli"
  chmod +x "${BUILD_DIR}/bin/whisper-cli"
  exit 0
fi
printf 'unexpected cmake command: %s\\n' "$*" >&2
exit 13
""",
    )


def test_prepare_outputs_sourceable_host_and_vlabor_env_blocks(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    binary = _write_executable(
        package.package_dir / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    completed = _run_prepare(
        package.preparer,
        ("--model-id", "base-q5_1", "--binary", str(binary)),
    )

    assert completed.returncode == 0
    assert "Trace file:" in completed.stderr
    env_file = tmp_path / "asr.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    model_path = package.package_dir / "models" / "whisper.cpp" / "ggml-base-q5_1.bin"
    host_model_path, host_worker_path, host_binary_path, trace_file = _source_env_block(env_file)
    assert (host_model_path, host_worker_path, host_binary_path) == (
        str(model_path.resolve(strict=True)),
        str(package.worker.resolve(strict=True)),
        str(binary.resolve(strict=True)),
    )
    assert Path(trace_file).is_file()
    assert Path(trace_file).parent == package.package_dir / "traces" / "prepare_whisper_cpp_asr"
    trace_text = Path(trace_file).read_text(encoding="utf-8")
    assert "started_at=" in trace_text
    assert "ended_at=" in trace_text
    assert "status=success" in trace_text
    assert "model_id=base-q5_1" in trace_text
    assert f"model_path={model_path.resolve(strict=True)}" in trace_text
    assert f"worker_path={package.worker.resolve(strict=True)}" in trace_text
    assert f"binary_path={binary.resolve(strict=True)}" in trace_text
    assert "build_requested=false" in trace_text
    assert "stage_start=" in trace_text
    assert "download_model" in trace_text
    assert "stage_finish=" in trace_text
    vlabor_values = _source_env_block(env_file, target="vlabor")
    assert vlabor_values[:3] == (
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_asr/models/whisper.cpp/ggml-base-q5_1.bin",
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_asr/scripts/whisper_cpp_worker",
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_asr/tools/whisper.cpp/build/bin/whisper-cli",
    )
    expected_container_trace = (
        "/ros2_ws/src/fluent_audio_ros2/"
        f"{Path(trace_file).relative_to(package.root_dir).as_posix()}"
    )
    assert vlabor_values[3] == expected_container_trace


def test_prepare_fails_closed_when_explicit_binary_is_missing(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    trace_file = tmp_path / "explicit-trace.log"
    missing_binary = tmp_path / "missing-whisper-cli"

    completed = _run_prepare(
        package.preparer,
        ("--trace-file", str(trace_file), "--binary", str(missing_binary)),
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert f"Trace file: {trace_file}" in completed.stderr
    real_reason = f"failure_reason=explicit --binary does not exist or is not a file: {missing_binary}"
    assert f"ERROR: explicit --binary does not exist or is not a file: {missing_binary}" in completed.stderr
    assert "explicit --binary is not executable: " not in completed.stderr
    trace_text = trace_file.read_text(encoding="utf-8")
    assert "status=failure" in trace_text
    assert "model_id=small-q5_1" in trace_text
    assert "model_path=" in trace_text
    assert "worker_path=" in trace_text
    assert "build_requested=false" in trace_text
    assert trace_text.count(real_reason) == 1
    assert "explicit --binary is not executable: " not in trace_text
    assert "failure_reason=command exited with status 1" not in trace_text


def test_prepare_uses_trace_file_from_environment(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    binary = _write_executable(
        package.package_dir / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    trace_file = tmp_path / "env-trace.log"

    completed = _run_prepare(
        package.preparer,
        ("--model-id", "tiny-q5_1", "--binary", str(binary)),
        env_updates={"FLUENT_AUDIO_ASR_PREPARE_TRACE_FILE": str(trace_file)},
    )

    assert completed.returncode == 0
    env_file = tmp_path / "asr-env-override.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    assert _source_env_block(env_file)[3] == str(trace_file)
    assert "status=success" in trace_file.read_text(encoding="utf-8")


def test_prepare_builds_package_local_whisper_cpp_with_fake_tools(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("bash", "basename", "chmod", "date", "dirname", "mkdir"):
        _link_command(bin_dir, command)
    _write_fake_git(bin_dir / "git")
    _write_fake_cmake(bin_dir / "cmake")

    completed = _run_prepare(
        package.preparer,
        ("--build-whisper-cpp",),
        env_updates={"PATH": str(bin_dir)},
    )

    assert completed.returncode == 0
    assert "Cloning whisper.cpp" in completed.stderr
    assert "Building whisper-cli" in completed.stderr
    env_file = tmp_path / "built.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    built_binary = (
        package.package_dir / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    )
    assert built_binary.is_file()
    assert _source_env_block(env_file)[2] == str(built_binary.resolve(strict=True))
