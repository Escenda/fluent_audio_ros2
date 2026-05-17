from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from fa_vad_py.backends.base import VADResult


logger = logging.getLogger(__name__)

VAD_THRESHOLD_START = 0.5
VAD_THRESHOLD_END = 0.1


class SileroVAD:
    """Streaming Silero VAD wrapper with hysteresis (offline-first)."""

    name = "silero"

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        hangover_ms: int = 250,
        threshold_start: float | None = None,
        threshold_end: float | None = None,
        model_path: str,
        execution_provider: str,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.threshold_start = (
            VAD_THRESHOLD_START
            if threshold_start is None
            else float(threshold_start)
        )
        self.threshold_end = (
            VAD_THRESHOLD_END if threshold_end is None else float(threshold_end)
        )
        self.hangover_frames = max(1, int(hangover_ms) // int(frame_ms))

        self._model_path = self._validate_model_path(model_path)
        self.device = self._validate_execution_provider(execution_provider)

        self._load_model()

        self._triggered = False
        self._hang_counter = 0
        self._required_samples = 512 if self.sample_rate == 16000 else 256
        self._min_window = self._required_samples
        self._target_window = self._required_samples
        self._max_window = max(self._target_window, int(self.sample_rate * 0.2))
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buf_ready = False
        self._warmup_logged = False

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
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return 0.0

        if self._buffer.size == 0:
            self._buffer = samples
        else:
            self._buffer = np.concatenate([self._buffer, samples])
        if self._buffer.size > self._max_window:
            self._buffer = self._buffer[-self._max_window :]

        if self._buffer.size < self._min_window:
            if not self._warmup_logged:
                logger.info(
                    "VAD warmup buffering... need>=%d have=%d",
                    self._min_window,
                    self._buffer.size,
                )
                self._warmup_logged = True
            return 0.0

        if not self._buf_ready:
            self._buf_ready = True
            logger.info("VAD buffer ready: size=%d -> start inference", self._buffer.size)

        window = self._buffer[-self._target_window :]
        if window.size > self._required_samples:
            window = window[-self._required_samples :]
        if window.size < self._required_samples:
            return 0.0

        tensor = torch.from_numpy(window).to(self.device)
        with torch.no_grad():
            probability = self.model(tensor, self.sample_rate).item()
        return float(probability)

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
    def _validate_execution_provider(execution_provider: str) -> torch.device:
        provider = execution_provider.strip()
        if not provider:
            raise RuntimeError("backend.execution_provider is required")
        if provider == "cpu":
            return torch.device("cpu")
        if provider == "cuda" or (
            provider.startswith("cuda:") and provider.removeprefix("cuda:").isdigit()
        ):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "backend.execution_provider requires CUDA but torch.cuda.is_available() is false"
                )
            if provider.startswith("cuda:"):
                device_index = int(provider.removeprefix("cuda:"))
                if device_index >= torch.cuda.device_count():
                    raise RuntimeError(
                        "backend.execution_provider CUDA device index is unavailable: "
                        f"{provider}"
                    )
            return torch.device(provider)
        raise RuntimeError(
            "unsupported backend.execution_provider for silero: "
            f"{provider}; supported providers: cpu, cuda, cuda:<index>"
        )

    def _load_model(self) -> None:
        logger.info("Loading Silero VAD model from local path: %s", self._model_path)
        self.model, _ = torch.hub.load(
            str(self._model_path),
            "silero_vad",
            source="local",
            trust_repo=True,
        )
        self.model.eval()
        self.model.to(self.device)
        logger.info("Silero VAD model loaded. device=%s", self.device)
