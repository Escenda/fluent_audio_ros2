from pathlib import Path

import pytest

from fa_asr_py.backends.local_command import (
    LocalCommandAsrBackend,
    load_external_worker_config,
    load_local_command_config,
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


def test_external_worker_requires_model_id(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.model is required"):
        load_external_worker_config(
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


def test_external_worker_backend_name_is_explicit(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    config = load_external_worker_config(
        command=str(command),
        model="gpt-4o-transcribe",
        language="ja",
        args=("--model", "{model}", "--audio", "{audio}"),
        timeout_sec=10.0,
        working_directory_value="",
        output_text_path="",
        workspace_dir=tmp_path / "work",
        cleanup_audio_files=True,
    )
    backend = LocalCommandAsrBackend(config, backend_name="openai_realtime")

    assert backend.name == "openai_realtime"
