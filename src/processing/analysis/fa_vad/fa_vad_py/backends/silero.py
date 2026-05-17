from __future__ import annotations

import logging
import os

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
        silero_repo_dir: str | None = None,
        allow_online: bool = False,
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

        self._silero_repo_dir = silero_repo_dir
        self._allow_online = bool(allow_online)

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

    def _load_model(self) -> None:
        logger.info("Loading Silero VAD model (offline-first)")

        repo_dir = self._silero_repo_dir.strip() if self._silero_repo_dir else ""
        repo_dir = os.path.expanduser(repo_dir)

        if repo_dir:
            if not os.path.isdir(repo_dir):
                raise RuntimeError(f"silero.repo_dir does not exist: {repo_dir}")
            self.model, _ = torch.hub.load(
                repo_dir,
                "silero_vad",
                source="local",
                trust_repo=True,
            )
            logger.info("Loaded Silero VAD from local repo: %s", repo_dir)
        elif self._allow_online:
            logger.warning("Loading Silero VAD from torch.hub online source")
            self.model, _ = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                trust_repo=True,
            )
        else:
            raise RuntimeError(
                "silero.repo_dir is required when silero.allow_online is false"
            )

        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        logger.info("Silero VAD model loaded. device=%s", self.device)
