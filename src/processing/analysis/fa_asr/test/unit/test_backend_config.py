from pathlib import Path

import pytest
import yaml

from fa_asr_py.backends.factory import AsrBackendSettings, build_asr_backend
from fa_asr_py.backends.local_command import LocalCommandAsrBackend, load_local_command_config
from fa_asr_py.backends.openai_realtime import (
    OpenAiRealtimeAsrBackend,
    load_openai_realtime_config,
)
from fa_asr_py.backends.parakeet_worker import (
    ParakeetWorkerAsrBackend,
    load_parakeet_worker_config,
)
from fa_asr_py.backends.whisper_cpp import WhisperCppAsrBackend, load_whisper_cpp_config


PACKAGE_ROOT = Path(__file__).parents[2]


def _settings(
    tmp_path: Path,
    *,
    backend_name: str,
    command: Path,
    model: str,
    model_path: str,
) -> AsrBackendSettings:
    return AsrBackendSettings(
        name=backend_name,
        command=str(command),
        model=model,
        model_path=model_path,
        language="ja",
        args=("--model", "{model}", "--audio", "{audio}"),
        timeout_sec=10.0,
        working_directory="",
        output_text_path="",
        workspace_dir=tmp_path / "work",
        cleanup_audio_files=True,
    )


def test_local_command_requires_existing_model_path(tmp_path: Path) -> None:
    command = tmp_path / "asr"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.model_path does not exist"):
        load_local_command_config(
            command=str(command),
            model_path_value=str(tmp_path / "missing.bin"),
            language="ja",
            args=("-m", "{model}", "-f", "{audio}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
        )


def test_default_config_requires_explicit_backend_name() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = PACKAGE_ROOT / "fa_asr_py" / "asr_node.py"
    source = source_path.read_text(encoding="utf-8")

    params = config["fa_asr"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["backend.args"] == []
    assert "ParameterUninitializedException" not in source
    assert "return tuple()" not in source


def test_asr_node_rejects_non_canonical_audio_frames() -> None:
    source_path = PACKAGE_ROOT / "fa_asr_py" / "asr_node.py"
    source = source_path.read_text(encoding="utf-8")

    assert "_resample_linear" not in source
    assert "_to_mono" not in source
    assert "np.frombuffer(bytes(msg.data), dtype=np.int16)" not in source
    assert "AudioFrame channels must be 1" in source
    assert "AudioFrame bit_depth must be 32" in source
    assert "AudioFrame sample_rate must match target_sample_rate" in source
    assert "AudioFrame samples must be normalized to [-1.0, 1.0]" in source


def test_command_backend_does_not_clip_audio() -> None:
    source_path = PACKAGE_ROOT / "fa_asr_py" / "backends" / "_command_process.py"
    source = source_path.read_text(encoding="utf-8")

    assert "np.clip" not in source
    assert "ASR request samples must be normalized to [-1.0, 1.0]" in source


def test_openai_realtime_requires_model_id(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.model is required"):
        load_openai_realtime_config(
            command=str(command),
            model="",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
        )


def test_parakeet_worker_requires_model_id(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.model is required"):
        load_parakeet_worker_config(
            command=str(command),
            model="",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
        )


def test_build_backend_requires_backend_name(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.name is required"):
        build_asr_backend(
            _settings(
                tmp_path,
                backend_name="",
                command=command,
                model="gpt-4o-transcribe",
                model_path="",
            )
        )


def test_build_backend_rejects_unknown_backend(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported ASR backend.name: bogus"):
        build_asr_backend(
            _settings(
                tmp_path,
                backend_name="bogus",
                command=command,
                model="gpt-4o-transcribe",
                model_path="",
            )
        )


def test_backends_use_dedicated_classes(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    local_command = build_asr_backend(
        _settings(
            tmp_path,
            backend_name="local_command",
            command=command,
            model="",
            model_path=str(model_path),
        )
    )
    whisper_cpp = build_asr_backend(
        _settings(
            tmp_path,
            backend_name="whisper_cpp",
            command=command,
            model="",
            model_path=str(model_path),
        )
    )
    parakeet_worker = build_asr_backend(
        _settings(
            tmp_path,
            backend_name="parakeet_worker",
            command=command,
            model="nvidia/parakeet",
            model_path="",
        )
    )
    openai_realtime = build_asr_backend(
        _settings(
            tmp_path,
            backend_name="openai_realtime",
            command=command,
            model="gpt-4o-realtime-preview",
            model_path="",
        )
    )

    assert isinstance(local_command, LocalCommandAsrBackend)
    assert isinstance(whisper_cpp, WhisperCppAsrBackend)
    assert isinstance(parakeet_worker, ParakeetWorkerAsrBackend)
    assert isinstance(openai_realtime, OpenAiRealtimeAsrBackend)


def test_whisper_cpp_uses_model_path_contract(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.model_path is required"):
        load_whisper_cpp_config(
            command=str(command),
            model_path_value="",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
        )
