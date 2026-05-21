from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


PACKAGE_ROOT = Path(__file__).parents[2]
PREPARER = PACKAGE_ROOT / "scripts" / "prepare_smart_turn_onnx"


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
    package_dir = root_dir / "src" / "ai" / "fa_turn_detector"
    script_dir = package_dir / "scripts"
    script_dir.mkdir(parents=True)
    (root_dir / "PRODUCT_OWNER_ROLE.md").write_text("test root marker\n", encoding="utf-8")
    (package_dir / "package.xml").write_text("<package></package>\n", encoding="utf-8")
    preparer = script_dir / PREPARER.name
    shutil.copy2(PREPARER, preparer)
    preparer.chmod(0o755)
    worker = _write_executable(script_dir / "smart_turn_onnx_worker", "#!/usr/bin/env bash\nexit 0\n")
    return PreparePackage(
        root_dir=root_dir,
        package_dir=package_dir,
        script_dir=script_dir,
        preparer=preparer,
        worker=worker,
    )


EXPECTED_SHA256 = "07a133aba31e2d0b523f17f8c2e4e65efe6d8f685efd12ca4fe21ebf4e798991"
EXPECTED_SIZE = "8757193"


def _write_model(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _run_prepare(
    script: Path,
    args: tuple[str, ...],
    *,
    env_updates: dict[str, str] | None = None,
    env_removals: tuple[str, ...] = ("FLUENT_AUDIO_TURN_DETECTOR_PREPARE_TRACE_FILE",),
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
                '"$FLUENT_AUDIO_TURN_DETECTOR_MODEL" '
                '"$FLUENT_AUDIO_TURN_DETECTOR_PROVIDER" '
                '"$FLUENT_AUDIO_TURN_DETECTOR_WORKER" '
                '"$FLUENT_AUDIO_TURN_DETECTOR_PREPARE_TRACE_FILE"'
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


def _write_fake_curl(path: Path, payload_file: Path) -> None:
    _write_executable(
        path,
        f"""#!/usr/bin/env bash
set -euo pipefail
OUTPUT_PATH=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${{OUTPUT_PATH}}" ]]; then
  printf 'missing --output\\n' >&2
  exit 14
fi
cp {str(payload_file)!r} "${{OUTPUT_PATH}}"
""",
    )


def _write_fake_wc(path: Path, size: str) -> None:
    _write_executable(
        path,
        f"""#!/usr/bin/env bash
set -euo pipefail
cat >/dev/null
printf '%s\\n' {size!r}
""",
    )


def _write_fake_sha256sum(path: Path, sha256: str) -> None:
    _write_executable(
        path,
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s  %s\\n' {sha256!r} "$1"
""",
    )


def _write_fake_validation_commands(bin_dir: Path, *, size: str, sha256: str) -> None:
    _write_fake_wc(bin_dir / "wc", size)
    _write_fake_sha256sum(bin_dir / "sha256sum", sha256)


def test_prepare_outputs_sourceable_host_and_vlabor_env_blocks(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    model_path = _write_model(
        package.package_dir / "models" / "smart-turn-v3" / "smart-turn-v3.0.onnx",
        b"fake smart turn onnx payload\n",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_validation_commands(bin_dir, size=EXPECTED_SIZE, sha256=EXPECTED_SHA256)

    completed = _run_prepare(
        package.preparer,
        ("--model-path", str(model_path)),
        env_updates={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert completed.returncode == 0
    assert "Trace file:" in completed.stderr
    env_file = tmp_path / "turn.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    host_model_path, host_provider, host_worker_path, trace_file = _source_env_block(env_file)
    assert (host_model_path, host_provider, host_worker_path) == (
        str(model_path.resolve(strict=True)),
        "CPUExecutionProvider",
        str(package.worker.resolve(strict=True)),
    )
    assert Path(trace_file).is_file()
    assert Path(trace_file).parent == package.package_dir / "traces" / "prepare_smart_turn_onnx"
    trace_text = Path(trace_file).read_text(encoding="utf-8")
    assert "started_at=" in trace_text
    assert "ended_at=" in trace_text
    assert "status=success" in trace_text
    assert f"model_path={model_path.resolve(strict=True)}" in trace_text
    assert "provider=CPUExecutionProvider" in trace_text
    assert f"model_sha256={EXPECTED_SHA256}" in trace_text
    assert f"model_size={EXPECTED_SIZE}" in trace_text
    assert f"worker_path={package.worker.resolve(strict=True)}" in trace_text
    assert "stage_start=" in trace_text
    assert "validate_model_file" in trace_text
    assert "stage_finish=" in trace_text

    vlabor_values = _source_env_block(env_file, target="vlabor")
    assert vlabor_values[:3] == (
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_turn_detector/models/smart-turn-v3/smart-turn-v3.0.onnx",
        "CPUExecutionProvider",
        "/ros2_ws/src/fluent_audio_ros2/src/ai/fa_turn_detector/scripts/smart_turn_onnx_worker",
    )
    expected_container_trace = (
        "/ros2_ws/src/fluent_audio_ros2/"
        f"{Path(trace_file).relative_to(package.root_dir).as_posix()}"
    )
    assert vlabor_values[3] == expected_container_trace


def test_prepare_downloads_missing_model_and_validates_payload(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    downloaded_payload = tmp_path / "smart-turn-v3.0.onnx"
    _write_model(downloaded_payload, b"downloaded onnx\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_curl(bin_dir / "curl", downloaded_payload)
    _write_fake_validation_commands(bin_dir, size=EXPECTED_SIZE, sha256=EXPECTED_SHA256)
    model_path = package.package_dir / "models" / "smart-turn-v3" / "smart-turn-v3.0.onnx"

    completed = _run_prepare(
        package.preparer,
        (
            "--model-path",
            str(model_path),
            "--model-url",
            "https://example.invalid/smart-turn-v3.0.onnx",
        ),
        env_updates={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert completed.returncode == 0
    assert "Downloading Smart Turn ONNX model" in completed.stderr
    assert model_path.read_bytes() == downloaded_payload.read_bytes()
    env_file = tmp_path / "downloaded-turn.env"
    env_file.write_text(completed.stdout, encoding="utf-8")
    assert _source_env_block(env_file)[:3] == (
        str(model_path.resolve(strict=True)),
        "CPUExecutionProvider",
        str(package.worker.resolve(strict=True)),
    )


def test_prepare_rejects_git_lfs_pointer_file(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    pointer_payload = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:07a133aba31e2d0b523f17f8c2e4e65efe6d8f685efd12ca4fe21ebf4e798991\n"
        b"size 8757193\n"
    )
    model_path = _write_model(
        package.package_dir / "models" / "smart-turn-v3" / "smart-turn-v3.0.onnx",
        pointer_payload,
    )
    trace_file = tmp_path / "pointer-trace.log"

    completed = _run_prepare(
        package.preparer,
        (
            "--trace-file",
            str(trace_file),
            "--model-path",
            str(model_path),
        ),
    )

    reason = f"Smart Turn ONNX model is a Git LFS pointer, not an ONNX payload: {model_path.resolve(strict=True)}"
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert f"ERROR: {reason}" in completed.stderr
    trace_text = trace_file.read_text(encoding="utf-8")
    assert "status=failure" in trace_text
    assert f"failure_reason={reason}" in trace_text
    assert "failure_reason=command exited with status 1" not in trace_text


def test_prepare_rejects_empty_model_file(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    model_path = package.package_dir / "models" / "smart-turn-v3" / "smart-turn-v3.0.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"")
    trace_file = tmp_path / "empty-trace.log"

    completed = _run_prepare(
        package.preparer,
        (
            "--trace-file",
            str(trace_file),
            "--model-path",
            str(model_path),
        ),
    )

    reason = f"Smart Turn ONNX model file is empty: {model_path.resolve(strict=True)}"
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert f"ERROR: {reason}" in completed.stderr
    trace_text = trace_file.read_text(encoding="utf-8")
    assert "status=failure" in trace_text
    assert f"failure_reason={reason}" in trace_text


def test_prepare_fails_closed_on_downloaded_size_mismatch(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    downloaded_payload = tmp_path / "smart-turn-v3.0.onnx"
    _write_model(downloaded_payload, b"downloaded onnx\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_curl(bin_dir / "curl", downloaded_payload)
    actual_size = str(int(EXPECTED_SIZE) + 1)
    _write_fake_validation_commands(bin_dir, size=actual_size, sha256=EXPECTED_SHA256)
    model_path = package.package_dir / "models" / "smart-turn-v3" / "smart-turn-v3.0.onnx"
    trace_file = tmp_path / "size-mismatch-trace.log"

    completed = _run_prepare(
        package.preparer,
        (
            "--trace-file",
            str(trace_file),
            "--model-path",
            str(model_path),
            "--model-url",
            "https://example.invalid/smart-turn-v3.0.onnx",
        ),
        env_updates={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    resolved_model_path = model_path.resolve(strict=True)
    reason = (
        f"Smart Turn ONNX model size mismatch: {resolved_model_path}; "
        f"expected {EXPECTED_SIZE} bytes, got {actual_size} bytes"
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert f"ERROR: {reason}" in completed.stderr
    trace_text = trace_file.read_text(encoding="utf-8")
    assert "status=failure" in trace_text
    assert f"failure_reason={reason}" in trace_text


def test_prepare_fails_closed_on_sha256_mismatch(tmp_path: Path) -> None:
    package = _copy_prepare_package(tmp_path)
    model_path = _write_model(
        package.package_dir / "models" / "smart-turn-v3" / "smart-turn-v3.0.onnx",
        b"fake smart turn onnx payload\n",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    actual_sha256 = "1" * 64
    _write_fake_validation_commands(bin_dir, size=EXPECTED_SIZE, sha256=actual_sha256)
    trace_file = tmp_path / "sha-mismatch-trace.log"

    completed = _run_prepare(
        package.preparer,
        (
            "--trace-file",
            str(trace_file),
            "--model-path",
            str(model_path),
        ),
        env_updates={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    resolved_model_path = model_path.resolve(strict=True)
    reason = (
        f"Smart Turn ONNX model sha256 mismatch: {resolved_model_path}; "
        f"expected {EXPECTED_SHA256}, got {actual_sha256}"
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert f"ERROR: {reason}" in completed.stderr
    trace_text = trace_file.read_text(encoding="utf-8")
    assert "status=failure" in trace_text
    assert f"failure_reason={reason}" in trace_text
