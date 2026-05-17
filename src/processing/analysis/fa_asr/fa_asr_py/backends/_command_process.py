from __future__ import annotations

import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fa_asr_py.backends.base import AsrRequest


@dataclass(frozen=True)
class CommandProcessConfig:
    executable: str
    model: str
    language: str
    args: tuple[str, ...]
    timeout_sec: float
    working_directory: Path | None
    output_text_path: str
    workspace_dir: Path
    cleanup_audio_files: bool


class _CommandProcessRunner:
    def __init__(self, config: CommandProcessConfig) -> None:
        self._config = config
        self._config.workspace_dir.mkdir(parents=True, exist_ok=True)

    def transcribe(self, request: AsrRequest) -> str:
        wav_path = self._config.workspace_dir / f"{time.time_ns()}_{request.user_turn_id}.wav"
        output_path = self._build_output_text_path(wav_path, request)
        try:
            self._write_wav(wav_path, request.samples, request.sample_rate)
            command = [self._config.executable]
            command.extend(self._format_args(wav_path, output_path))
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self._config.working_directory)
                    if self._config.working_directory is not None
                    else None,
                    capture_output=True,
                    text=True,
                    timeout=self._config.timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("ASR backend command timed out") from exc
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                raise RuntimeError(
                    f"ASR backend command failed: code={completed.returncode} stderr={stderr}"
                )

            transcript = self._read_transcript(completed.stdout, output_path)
            if not transcript:
                raise RuntimeError("ASR backend returned an empty transcript")
            return transcript
        finally:
            if self._config.cleanup_audio_files and wav_path.exists():
                wav_path.unlink()

    def _build_output_text_path(self, wav_path: Path, request: AsrRequest) -> Path | None:
        if not self._config.output_text_path:
            return None
        rendered = self._config.output_text_path.format(
            audio=str(wav_path),
            model=self._config.model,
            language=self._config.language,
            session_id=request.session_id,
            user_turn_id=request.user_turn_id,
        )
        path = Path(rendered).expanduser()
        if not path.is_absolute():
            path = self._config.workspace_dir / path
        return path

    def _format_args(self, wav_path: Path, output_path: Path | None) -> list[str]:
        output_value = "" if output_path is None else str(output_path)
        return [
            part.format(
                audio=str(wav_path),
                model=self._config.model,
                language=self._config.language,
                output=output_value,
            )
            for part in self._config.args
        ]

    @staticmethod
    def _read_transcript(stdout_text: str, output_path: Path | None) -> str:
        if output_path is not None:
            if not output_path.is_file():
                raise RuntimeError(f"ASR output text file was not created: {output_path}")
            return output_path.read_text(encoding="utf-8").strip()
        return stdout_text.strip()

    @staticmethod
    def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
        pcm = _float_to_pcm16(samples)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)


def _load_model_path_command_config(
    *,
    command: str,
    model_path_value: str,
    language: str,
    args: tuple[str, ...],
    timeout_sec: float,
    working_directory_value: str,
    output_text_path: str,
    workspace_dir: Path,
    cleanup_audio_files: bool,
) -> CommandProcessConfig:
    if not model_path_value:
        raise RuntimeError("backend.model_path is required")
    model_path = Path(model_path_value).expanduser()
    if not model_path.is_file():
        raise RuntimeError(f"backend.model_path does not exist: {model_path}")
    return _load_command_config(
        command=command,
        model_value=str(model_path),
        language=language,
        args=args,
        timeout_sec=timeout_sec,
        working_directory_value=working_directory_value,
        output_text_path=output_text_path,
        workspace_dir=workspace_dir,
        cleanup_audio_files=cleanup_audio_files,
    )


def _load_model_id_command_config(
    *,
    command: str,
    model: str,
    language: str,
    args: tuple[str, ...],
    timeout_sec: float,
    working_directory_value: str,
    output_text_path: str,
    workspace_dir: Path,
    cleanup_audio_files: bool,
) -> CommandProcessConfig:
    model_value = model.strip()
    if not model_value:
        raise RuntimeError("backend.model is required")
    return _load_command_config(
        command=command,
        model_value=model_value,
        language=language,
        args=args,
        timeout_sec=timeout_sec,
        working_directory_value=working_directory_value,
        output_text_path=output_text_path,
        workspace_dir=workspace_dir,
        cleanup_audio_files=cleanup_audio_files,
    )


def _load_command_config(
    *,
    command: str,
    model_value: str,
    language: str,
    args: tuple[str, ...],
    timeout_sec: float,
    working_directory_value: str,
    output_text_path: str,
    workspace_dir: Path,
    cleanup_audio_files: bool,
) -> CommandProcessConfig:
    executable = _resolve_executable(command)

    if not args:
        raise RuntimeError("backend.args must not be empty")
    joined_args = " ".join(args)
    if "{audio}" not in joined_args:
        raise RuntimeError("backend.args must include the {audio} placeholder")
    if "{model}" not in joined_args:
        raise RuntimeError("backend.args must include the {model} placeholder")

    if timeout_sec <= 0.0:
        raise RuntimeError("backend.timeout_sec must be greater than zero")

    working_directory = (
        Path(working_directory_value).expanduser() if working_directory_value else None
    )
    if working_directory is not None and not working_directory.is_dir():
        raise RuntimeError(f"backend.working_directory does not exist: {working_directory}")

    return CommandProcessConfig(
        executable=executable,
        model=model_value,
        language=language,
        args=args,
        timeout_sec=timeout_sec,
        working_directory=working_directory,
        output_text_path=output_text_path,
        workspace_dir=workspace_dir,
        cleanup_audio_files=cleanup_audio_files,
    )


def _resolve_executable(command: str) -> str:
    if not command:
        raise RuntimeError("backend.command is required")
    command_path = Path(command).expanduser()
    if "/" in command or command_path.is_absolute():
        if not command_path.is_file():
            raise RuntimeError(f"backend.command does not exist: {command_path}")
        return str(command_path)
    resolved = shutil.which(command)
    if not resolved:
        raise RuntimeError(f"backend.command not found in PATH: {command}")
    return resolved


def _float_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()
