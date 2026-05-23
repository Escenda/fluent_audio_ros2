from __future__ import annotations

from dataclasses import dataclass

from fa_vad_py.vad_probability_stream import VadProbabilityFrame


@dataclass(frozen=True)
class SpeechActivityConfig:
    speech_threshold: float = 0.55
    silence_threshold: float = 0.35
    start_delta: float = 0.20
    end_delta: float = 0.20
    min_start_probability: float = 0.35
    max_end_probability: float = 0.45
    smoothing_alpha: float = 0.35
    start_consecutive_windows: int = 2
    end_consecutive_windows: int = 4
    min_speech_ms: int = 120
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        for name in (
            "speech_threshold",
            "silence_threshold",
            "start_delta",
            "end_delta",
            "min_start_probability",
            "max_end_probability",
            "smoothing_alpha",
        ):
            value = getattr(self, name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0]")
        if self.silence_threshold > self.speech_threshold:
            raise ValueError("silence_threshold must be <= speech_threshold")
        if self.start_consecutive_windows <= 0:
            raise ValueError("start_consecutive_windows must be > 0")
        if self.end_consecutive_windows <= 0:
            raise ValueError("end_consecutive_windows must be > 0")
        if self.min_speech_ms < 0:
            raise ValueError("min_speech_ms must be >= 0")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")


@dataclass(frozen=True)
class VoiceActivitySnapshot:
    probability: float
    smoothed_probability: float
    is_speech: bool
    speech_started: bool
    speech_ended: bool
    window_start_sample: int
    window_end_sample: int


class SpeechActivityEstimator:
    """Infers speech state from VAD probability frames."""

    def __init__(self, config: SpeechActivityConfig) -> None:
        self.config = config
        self._min_speech_samples = int(config.sample_rate * config.min_speech_ms / 1000)
        self.reset()

    def reset(self) -> None:
        self._smoothed_probability = 0.0
        self._last_probability = 0.0
        self._is_speech = False
        self._speech_start_sample = 0
        self._start_candidates = 0
        self._end_candidates = 0

    def update(self, frame: VadProbabilityFrame) -> VoiceActivitySnapshot:
        probability = max(0.0, min(1.0, float(frame.probability)))
        self._smoothed_probability = self._smooth(probability)
        speech_started = self._maybe_start_speech(
            probability,
            frame.window_start_sample,
        )
        speech_ended = self._maybe_end_speech(probability, frame.window_end_sample)
        self._last_probability = probability
        return VoiceActivitySnapshot(
            probability=probability,
            smoothed_probability=self._smoothed_probability,
            is_speech=self._is_speech,
            speech_started=speech_started,
            speech_ended=speech_ended,
            window_start_sample=frame.window_start_sample,
            window_end_sample=frame.window_end_sample,
        )

    def _smooth(self, probability: float) -> float:
        alpha = self.config.smoothing_alpha
        return (alpha * probability) + ((1.0 - alpha) * self._smoothed_probability)

    def _maybe_start_speech(self, probability: float, start_sample: int) -> bool:
        if self._is_speech:
            self._start_candidates = 0
            return False

        rising = probability - self._last_probability
        candidate = probability >= self.config.speech_threshold or (
            probability >= self.config.min_start_probability
            and rising >= self.config.start_delta
        )
        self._start_candidates = self._start_candidates + 1 if candidate else 0
        if self._start_candidates < self.config.start_consecutive_windows:
            return False

        self._is_speech = True
        self._end_candidates = 0
        self._speech_start_sample = start_sample
        return True

    def _maybe_end_speech(self, probability: float, end_sample: int) -> bool:
        if not self._is_speech:
            self._end_candidates = 0
            return False

        falling = self._last_probability - probability
        candidate = probability <= self.config.silence_threshold or (
            probability <= self.config.max_end_probability
            and falling >= self.config.end_delta
        )
        self._end_candidates = self._end_candidates + 1 if candidate else 0
        speech_samples = end_sample - self._speech_start_sample
        if self._end_candidates < self.config.end_consecutive_windows:
            return False
        if speech_samples < self._min_speech_samples:
            return False

        self._is_speech = False
        self._start_candidates = 0
        return True
