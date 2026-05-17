from __future__ import annotations

import os
import shutil
import subprocess
import time
import wave
from pathlib import Path

from fa_vad_py.backends.base import VADResult


class SileroVAD:
    """External-process Silero VAD adapter.

    The ROS2 node owns topic/message conversion. The Silero runtime runs behind
    an explicit command boundary so PyTorch/Python-version requirements do not
    leak into this node process.
    """

    name = "silero"

    def __init__(
        self,
        *,
        sample_rate: int,
        frame_ms: int,
        hangover_ms: int,
        threshold_start: float,
        threshold_end: float,
        model_path: str,
        execution_provider: str,
        command: str,
        args: tuple[str, ...],
        timeout_sec: float,
        workspace_dir: str,
        cleanup_audio_files: bool,
    ) -> None:
        self.sample_rate = self._validate_sample_rate(sample_rate)
        self.frame_ms = self._validate_frame_ms(frame_ms)
        self.threshold_start, self.threshold_end = self._validate_thresholds(
            threshold_start=threshold_start,
            threshold_end=threshold_end,
        )
        self.hangover_frames = self._validate_hangover_frames(
            hangover_ms=hangover_ms,
            frame_ms=self.frame_ms,
        )

        self._model_path = self._validate_model_path(model_path)
        self._execution_provider = self._validate_execution_provider(execution_provider)
        self._command = self._validate_command(command)
        self._args = self._validate_args(args)
        self._timeout_sec = self._validate_timeout(timeout_sec)
        self._workspace_dir = self._validate_workspace_dir(workspace_dir)
        self._cleanup_audio_files = bool(cleanup_audio_files)

        self._triggered = False
        self._hang_counter = 0
        self._required_samples = 512 if self.sample_rate == 16000 else 256
        self._max_samples = max(self._required_samples, int(self.sample_rate * 0.2))
        self._buffer = bytearray()
        self._buf_ready = False

    def update(self, pcm16_bytes: bytes) -> VADResult:
        probability = self._probability(pcm16_bytes)
        is_speech = probability >= self.threshold_start

        start = False
        end = False

        if not self._triggered:
            if is_speech:
                self._triggered = True
                self._hang_counter = self.hangover_frames
                start = True
        else:
            if probability < self.threshold_end:
                self._hang_counter -= 1
                if self._hang_counter <= 0:
                    self._triggered = False
                    end = True
            else:
                self._hang_counter = self.hangover_frames

        return VADResult(probability, self._triggered, start, end)

    def _probability(self, pcm16_bytes: bytes) -> float:
        if not pcm16_bytes:
            return 0.0

        self._buffer.extend(pcm16_bytes)
        max_bytes = self._max_samples * 2
        if len(self._buffer) > max_bytes:
            self._buffer = self._buffer[-max_bytes:]

        required_bytes = self._required_samples * 2
        if len(self._buffer) < required_bytes:
            return 0.0

        self._buf_ready = True
        window = bytes(self._buffer[-required_bytes:])
        wav_path = self._workspace_dir / f"{time.time_ns()}_vad.wav"
        try:
            self._write_wav(wav_path, window)
            command = [self._command]
            command.extend(self._format_args(wav_path))
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("Silero VAD backend command timed out") from exc
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                raise RuntimeError(
                    "Silero VAD backend command failed: "
                    f"code={completed.returncode} stderr={stderr}"
                )
            return self._parse_probability(completed.stdout)
        finally:
            if self._cleanup_audio_files and wav_path.exists():
                wav_path.unlink()

    def _format_args(self, wav_path: Path) -> list[str]:
        return [
            item.format(
                audio=str(wav_path),
                model=str(self._model_path),
                provider=self._execution_provider,
                sample_rate=str(self.sample_rate),
            )
            for item in self._args
        ]

    def _write_wav(self, wav_path: Path, pcm16_bytes: bytes) -> None:
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm16_bytes)

    @staticmethod
    def _validate_sample_rate(sample_rate: int) -> int:
        sample_rate_value = int(sample_rate)
        if sample_rate_value not in (8000, 16000):
            raise RuntimeError("sample_rate must be 8000 or 16000 for silero")
        return sample_rate_value

    @staticmethod
    def _validate_frame_ms(frame_ms: int) -> int:
        frame_ms_value = int(frame_ms)
        if frame_ms_value <= 0:
            raise RuntimeError("frame_ms must be > 0")
        return frame_ms_value

    @staticmethod
    def _validate_thresholds(
        *,
        threshold_start: float,
        threshold_end: float,
    ) -> tuple[float, float]:
        start = float(threshold_start)
        end = float(threshold_end)
        if start < 0.0 or start > 1.0:
            raise RuntimeError("threshold_start must be in [0.0, 1.0]")
        if end < 0.0 or end > 1.0:
            raise RuntimeError("threshold_end must be in [0.0, 1.0]")
        if end > start:
            raise RuntimeError("threshold_end must be <= threshold_start")
        return start, end

    @staticmethod
    def _validate_hangover_frames(*, hangover_ms: int, frame_ms: int) -> int:
        hangover_ms_value = int(hangover_ms)
        if hangover_ms_value <= 0:
            raise RuntimeError("hangover_ms must be > 0")
        if hangover_ms_value < frame_ms:
            raise RuntimeError("hangover_ms must be >= frame_ms")
        if hangover_ms_value % frame_ms != 0:
            raise RuntimeError("hangover_ms must be divisible by frame_ms")
        return hangover_ms_value // frame_ms

    @staticmethod
    def _validate_model_path(model_path: str) -> Path:
        path_value = model_path.strip()
        if not path_value:
            raise RuntimeError("backend.model_path is required")
        path = Path(path_value).expanduser()
        if not path.is_dir():
            raise RuntimeError(f"backend.model_path does not exist: {path}")
        return path

    @staticmethod
    def _validate_execution_provider(execution_provider: str) -> str:
        provider = execution_provider.strip()
        if not provider:
            raise RuntimeError("backend.execution_provider is required")
        if provider == "cpu":
            return provider
        if provider == "cuda":
            return provider
        if provider.startswith("cuda:") and provider.removeprefix("cuda:").isdigit():
            return provider
        raise RuntimeError(
            "unsupported backend.execution_provider for silero: "
            f"{provider}; supported providers: cpu, cuda, cuda:<index>"
        )

    @staticmethod
    def _validate_command(command: str) -> str:
        command_value = command.strip()
        if not command_value:
            raise RuntimeError("backend.command is required")
        if "/" in command_value:
            command_path = Path(command_value).expanduser()
            if not command_path.is_file() or not os.access(command_path, os.X_OK):
                raise RuntimeError(f"backend.command is not executable: {command_path}")
            return str(command_path)
        resolved = shutil.which(command_value)
        if resolved is None:
            raise RuntimeError(f"backend.command not found on PATH: {command_value}")
        return resolved

    @staticmethod
    def _validate_args(args: tuple[str, ...]) -> tuple[str, ...]:
        if not args:
            raise RuntimeError("backend.args is required")
        rendered = " ".join(args)
        for placeholder in ("{audio}", "{model}", "{provider}", "{sample_rate}"):
            if placeholder not in rendered:
                raise RuntimeError(f"backend.args must include the {placeholder} placeholder")
        return args

    @staticmethod
    def _validate_timeout(timeout_sec: float) -> float:
        if timeout_sec <= 0.0:
            raise RuntimeError("backend.timeout_sec must be > 0")
        return timeout_sec

    @staticmethod
    def _validate_workspace_dir(workspace_dir: str) -> Path:
        path_value = workspace_dir.strip()
        if not path_value:
            raise RuntimeError("backend.workspace_dir is required")
        path = Path(path_value).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _parse_probability(stdout_text: str) -> float:
        values = [line.strip() for line in stdout_text.splitlines() if line.strip()]
        if not values:
            raise RuntimeError("Silero VAD backend command returned empty stdout")
        try:
            probability = float(values[-1])
        except ValueError as exc:
            raise RuntimeError(
                "Silero VAD backend command must print probability as a float"
            ) from exc
        if probability < 0.0 or probability > 1.0:
            raise RuntimeError("Silero VAD backend probability must be in [0.0, 1.0]")
        return probability
