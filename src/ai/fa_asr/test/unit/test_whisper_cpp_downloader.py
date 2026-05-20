from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


PACKAGE_ROOT = Path(__file__).parents[2]
DOWNLOADER = PACKAGE_ROOT / "scripts" / "download_whisper_cpp_model"
WORKER = PACKAGE_ROOT / "scripts" / "whisper_cpp_worker"
SMALL_Q5_1_SHA1 = "6fe57ddcfdd1c6b07cdcc73aaf620810ce5fc771"
BASE_Q5_1_SHA1 = "a3733eda680ef76256db5fc5dd9de8629e62c5e7"
WRONG_SHA1 = "0000000000000000000000000000000000000000"


@dataclass(frozen=True)
class DownloaderPackage:
    script: Path
    worker: Path
    package_dir: Path


def _copy_source_package(tmp_path: Path) -> DownloaderPackage:
    package_dir = tmp_path / "source" / "fa_asr"
    script_dir = package_dir / "scripts"
    script_dir.mkdir(parents=True)
    (package_dir / "package.xml").write_text("<package></package>\n", encoding="utf-8")
    return _copy_scripts(script_dir=script_dir, package_dir=package_dir)


def _copy_installed_package(tmp_path: Path) -> DownloaderPackage:
    script_dir = tmp_path / "install" / "fa_asr" / "lib" / "fa_asr"
    script_dir.mkdir(parents=True)
    return _copy_scripts(script_dir=script_dir, package_dir=script_dir.parent)


def _copy_scripts(*, script_dir: Path, package_dir: Path) -> DownloaderPackage:
    script = script_dir / DOWNLOADER.name
    worker = script_dir / WORKER.name
    shutil.copy2(DOWNLOADER, script)
    shutil.copy2(WORKER, worker)
    script.chmod(0o755)
    worker.chmod(0o755)
    return DownloaderPackage(script=script, worker=worker, package_dir=package_dir)


def _write_fake_sha1sum(bin_dir: Path, sha1_value: str) -> Path:
    bin_dir.mkdir(parents=True)
    executable = bin_dir / "sha1sum"
    executable.write_text(
        f"#!/usr/bin/env bash\nprintf '%s  %s\\n' '{sha1_value}' \"$1\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _write_model_file(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fake model checkpoint")
    return path


def _run_downloader(
    script: Path,
    args: tuple[str, ...] = (),
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


def test_downloader_rejects_unsupported_model_without_download(tmp_path: Path) -> None:
    completed = _run_downloader(
        DOWNLOADER,
        ("small.en-tdrz", str(tmp_path / "models")),
    )

    assert completed.returncode == 2
    assert "unsupported whisper.cpp model id: small.en-tdrz" in completed.stderr
    assert "FLUENT_AUDIO_ASR_MODEL_PATH" not in completed.stdout


def test_downloader_accepts_existing_checksum_and_prints_worker_path(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models"
    model_path = _write_model_file(model_dir / "ggml-base-q5_1.bin")
    fake_bin = tmp_path / "bin"
    _write_fake_sha1sum(fake_bin, BASE_Q5_1_SHA1)

    completed = _run_downloader(
        DOWNLOADER,
        ("base-q5_1", str(model_dir)),
        {"PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.splitlines() == [
        f"FLUENT_AUDIO_ASR_MODEL_PATH={model_path.resolve(strict=True)}",
        f"FLUENT_AUDIO_ASR_WORKER={WORKER.resolve(strict=True)}",
    ]


def test_downloader_rejects_existing_checksum_mismatch(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_path = _write_model_file(model_dir / "ggml-small-q5_1.bin")
    fake_bin = tmp_path / "bin"
    _write_fake_sha1sum(fake_bin, WRONG_SHA1)

    completed = _run_downloader(
        DOWNLOADER,
        ("small-q5_1", str(model_dir)),
        {"PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
    )

    assert completed.returncode == 1
    assert "existing model checksum mismatch" in completed.stderr
    assert str(model_path.resolve(strict=True)) in completed.stderr
    assert "FLUENT_AUDIO_ASR_MODEL_PATH" not in completed.stdout


def test_downloader_source_default_uses_package_models_directory(tmp_path: Path) -> None:
    package = _copy_source_package(tmp_path)
    model_path = _write_model_file(
        package.package_dir / "models" / "whisper.cpp" / "ggml-small-q5_1.bin"
    )
    fake_bin = tmp_path / "bin"
    _write_fake_sha1sum(fake_bin, SMALL_Q5_1_SHA1)

    completed = _run_downloader(
        package.script,
        env_updates={"PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.splitlines() == [
        f"FLUENT_AUDIO_ASR_MODEL_PATH={model_path.resolve(strict=True)}",
        f"FLUENT_AUDIO_ASR_WORKER={package.worker.resolve(strict=True)}",
    ]


def test_downloader_installed_default_uses_home_cache(tmp_path: Path) -> None:
    package = _copy_installed_package(tmp_path)
    home_dir = tmp_path / "home"
    model_path = _write_model_file(
        home_dir
        / ".cache"
        / "fluent_audio"
        / "fa_asr"
        / "models"
        / "whisper.cpp"
        / "ggml-small-q5_1.bin"
    )
    fake_bin = tmp_path / "bin"
    _write_fake_sha1sum(fake_bin, SMALL_Q5_1_SHA1)

    completed = _run_downloader(
        package.script,
        env_updates={
            "HOME": str(home_dir),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.splitlines() == [
        f"FLUENT_AUDIO_ASR_MODEL_PATH={model_path.resolve(strict=True)}",
        f"FLUENT_AUDIO_ASR_WORKER={package.worker.resolve(strict=True)}",
    ]
