from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from fa_asr_py.backends.base import (
    ASR_AUDIO_ENCODING_FLOAT32LE,
    AsrAudioPayload,
    AsrBackendCapability,
    AsrRequest,
    AsrStreamRequest,
    AsrStreamResult,
    AsrTranscript,
    AsrTranscriptSegment,
    build_asr_transcript,
    plain_text_to_asr_transcript,
)

_PARAKEET_SAMPLE_RATE_HZ = 16000
_PARAKEET_CHANNELS = 1
_SUPPORTED_MODEL_MARKERS = ("parakeet", "1.1b")
_LANGUAGE_POLICY_AUTO_DETECT = "auto_detect"
_DEFAULT_CHUNK_SIZE_SAMPLES = 1600
_DEFAULT_MAX_BUFFER_SEC = 30.0
_DEFAULT_SPEECH_ENERGY_THRESHOLD = 0.001


@dataclass(frozen=True)
class ParakeetMultilingualBufferedConfig:
    model: str
    model_path: Path | None
    language: str
    language_policy: str
    sample_rate_hz: int
    channels: int
    chunk_size_samples: int
    emit_partial: bool
    max_buffer_samples: int
    speech_energy_threshold: float


class _TranscriptionHypothesis(Protocol):
    text: str


class _ParakeetRunner(Protocol):
    def transcribe(self, samples: np.ndarray) -> str:
        ...


class _NeMoAsrModel(Protocol):
    def transcribe(
        self,
        audio: Sequence[np.ndarray],
        *,
        batch_size: int,
        return_hypotheses: bool,
        num_workers: int,
        verbose: bool,
    ) -> Sequence[str | _TranscriptionHypothesis]:
        ...


class _ParakeetRunnerClass(Protocol):
    def __call__(self, config: ParakeetMultilingualBufferedConfig) -> _ParakeetRunner:
        ...


class ParakeetMultilingualBufferedAsrBackend:
    name = "parakeet_multilingual_buffered"

    def __init__(
        self,
        config: ParakeetMultilingualBufferedConfig,
        *,
        runner_class: _ParakeetRunnerClass | None = None,
    ) -> None:
        self._config = config
        self._runner = (runner_class or _ParakeetMultilingualBufferedRunner)(config)
        self.capability = _capability(config)

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        request.payload.validate_matches(self.capability)
        text = self._runner.transcribe(request.payload.float32_samples())
        return plain_text_to_asr_transcript(text, sample_count=request.payload.sample_count)

    def start_stream(self, request: AsrStreamRequest) -> "ParakeetMultilingualBufferedSession":
        _validate_stream_request(request)
        return ParakeetMultilingualBufferedSession(self._runner, self._config)


class ParakeetMultilingualBufferedSession:
    def __init__(
        self,
        runner: _ParakeetRunner,
        config: ParakeetMultilingualBufferedConfig,
    ) -> None:
        self._runner = runner
        self._config = config
        self._buffer = np.array([], dtype=np.float32)
        self._absolute_sample_count = 0
        self._next_decode_sample_count = config.chunk_size_samples
        self._last_partial_text = ""
        self._finished = False
        self._cancelled = False

    def push_audio(self, payload: AsrAudioPayload) -> tuple[AsrStreamResult, ...]:
        self._ensure_open()
        payload.validate_matches(_capability(self._config))
        samples = payload.float32_samples()
        self._append_samples(samples)
        if not self._config.emit_partial:
            return ()

        results: list[AsrStreamResult] = []
        while self._absolute_sample_count >= self._next_decode_sample_count:
            partial = self._decode(is_final=False)
            if partial is not None:
                results.append(partial)
            self._next_decode_sample_count += self._config.chunk_size_samples
        return tuple(results)

    def drain_results(self) -> tuple[AsrStreamResult, ...]:
        self._ensure_open()
        return ()

    def finish(self) -> tuple[AsrStreamResult, ...]:
        self._ensure_open()
        if self._buffer.size == 0:
            raise RuntimeError("ASR stream finish requires buffered audio")
        final = self._decode(is_final=True)
        if final is None:
            raise RuntimeError("ASR final transcript must not be empty for speech audio")
        self._finished = True
        return (final,)

    def cancel(self) -> None:
        self._buffer = np.array([], dtype=np.float32)
        self._cancelled = True

    def _append_samples(self, samples: np.ndarray) -> None:
        if self._buffer.size == 0:
            self._buffer = samples.copy()
        else:
            self._buffer = np.concatenate((self._buffer, samples))
        self._absolute_sample_count += int(samples.size)
        if self._buffer.size > self._config.max_buffer_samples:
            self._buffer = self._buffer[-self._config.max_buffer_samples :]

    def _decode(self, *, is_final: bool) -> AsrStreamResult | None:
        text = self._runner.transcribe(self._buffer).strip()
        if not text:
            if is_final and not _speech_energy_is_sufficient(
                self._buffer,
                self._config.speech_energy_threshold,
            ):
                return _stream_result_from_text(
                    text,
                    sample_count=self._absolute_sample_count,
                    is_final=True,
                    allow_empty_text=True,
                )
            if is_final:
                raise RuntimeError("ASR final transcript must not be empty for speech audio")
            return None
        if not is_final and text == self._last_partial_text:
            return None
        if not is_final:
            self._last_partial_text = text
        return _stream_result_from_text(
            text,
            sample_count=self._absolute_sample_count,
            is_final=is_final,
            allow_empty_text=False,
        )

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError("ASR stream is already finished")
        if self._cancelled:
            raise RuntimeError("ASR stream is already cancelled")


class _ParakeetMultilingualBufferedRunner:
    def __init__(self, config: ParakeetMultilingualBufferedConfig) -> None:
        self._model = self._load_model(config)

    def transcribe(self, samples: np.ndarray) -> str:
        _validate_samples_for_inference(samples)
        results = self._model.transcribe(
            [samples],
            batch_size=1,
            return_hypotheses=True,
            num_workers=0,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError("Parakeet ASR backend returned an unexpected result count")
        return _transcription_text(results[0])

    def _load_model(self, config: ParakeetMultilingualBufferedConfig) -> _NeMoAsrModel:
        # NeMo is imported only when this backend is instantiated so unit tests can
        # exercise config/session behavior without importing the model runtime.
        from nemo.collections.asr.models import ASRModel

        if config.model_path is not None:
            return ASRModel.restore_from(str(config.model_path))
        return ASRModel.from_pretrained(model_name=config.model)


def load_parakeet_multilingual_buffered_config(
    *,
    model: str,
    model_path_value: str,
    language: str,
    sample_rate_hz: int,
    channels: int,
    language_policy: str = _LANGUAGE_POLICY_AUTO_DETECT,
    chunk_size_samples: int = _DEFAULT_CHUNK_SIZE_SAMPLES,
    chunk_ms: int = 0,
    emit_partial: bool = True,
    max_buffer_sec: float = _DEFAULT_MAX_BUFFER_SEC,
    speech_energy_threshold: float = _DEFAULT_SPEECH_ENERGY_THRESHOLD,
) -> ParakeetMultilingualBufferedConfig:
    model_value = model.strip()
    model_path = _resolve_optional_model_path(model_path_value)
    if not model_value and model_path is None:
        raise RuntimeError("backend.model or backend.model_path is required")
    if model_value and model_path is not None:
        raise RuntimeError("set only one of backend.model or backend.model_path")
    if model_value:
        _validate_multilingual_parakeet_model(model_value)
    if model_path is not None:
        _validate_multilingual_parakeet_model(str(model_path))
    language_value = language.strip()
    policy_value = language_policy.strip()
    if policy_value != _LANGUAGE_POLICY_AUTO_DETECT:
        raise RuntimeError("backend.language_policy must be auto_detect")
    if language_value:
        raise RuntimeError(
            "backend.language is not supported by parakeet_multilingual_buffered; "
            "set backend.language_policy=auto_detect"
        )
    if sample_rate_hz != _PARAKEET_SAMPLE_RATE_HZ:
        raise RuntimeError(
            "backend.sample_rate_hz must be 16000 for parakeet_multilingual_buffered"
        )
    if channels != _PARAKEET_CHANNELS:
        raise RuntimeError("backend.channels must be 1 for parakeet_multilingual_buffered")
    if type(emit_partial) is not bool:
        raise RuntimeError("backend.emit_partial must be a bool")
    decode_chunk_size = _decode_chunk_size(
        sample_rate_hz=sample_rate_hz,
        chunk_size_samples=chunk_size_samples,
        chunk_ms=chunk_ms,
        emit_partial=emit_partial,
    )
    buffer_samples = _max_buffer_samples(
        sample_rate_hz=sample_rate_hz,
        max_buffer_sec=max_buffer_sec,
    )
    energy_threshold = _speech_energy_threshold(speech_energy_threshold)
    return ParakeetMultilingualBufferedConfig(
        model=model_value,
        model_path=model_path,
        language=language_value,
        language_policy=policy_value,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        chunk_size_samples=decode_chunk_size,
        emit_partial=emit_partial,
        max_buffer_samples=buffer_samples,
        speech_energy_threshold=energy_threshold,
    )


def _capability(config: ParakeetMultilingualBufferedConfig) -> AsrBackendCapability:
    return AsrBackendCapability(
        audio_encoding=ASR_AUDIO_ENCODING_FLOAT32LE,
        sample_rate_hz=_PARAKEET_SAMPLE_RATE_HZ,
        channels=_PARAKEET_CHANNELS,
        streaming=True,
        final_results_only=not config.emit_partial,
    )


def _resolve_optional_model_path(model_path_value: str) -> Path | None:
    if not model_path_value.strip():
        return None
    model_path = Path(model_path_value).expanduser()
    try:
        resolved = model_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"backend.model_path does not exist: {model_path}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"backend.model_path is not a file: {resolved}")
    return resolved


def _validate_multilingual_parakeet_model(model: str) -> None:
    normalized = model.lower()
    if "nemotron" in normalized:
        raise RuntimeError("backend.model must not use English-only Nemotron ASR")
    for marker in _SUPPORTED_MODEL_MARKERS:
        if marker not in normalized:
            raise RuntimeError(
                "backend.model must identify a multilingual Parakeet 1.1B model"
            )


def _validate_samples_for_inference(samples: np.ndarray) -> None:
    if samples.dtype != np.float32:
        raise RuntimeError("Parakeet ASR samples must be float32")
    if samples.ndim != 1:
        raise RuntimeError("Parakeet ASR samples must be one-dimensional")
    if samples.size <= 0:
        raise RuntimeError("Parakeet ASR samples are required")
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("Parakeet ASR samples must be finite")


def _decode_chunk_size(
    *,
    sample_rate_hz: int,
    chunk_size_samples: int,
    chunk_ms: int,
    emit_partial: bool,
) -> int:
    if type(chunk_size_samples) is not int:
        raise RuntimeError("backend.chunk_size_samples must be an integer")
    if type(chunk_ms) is not int:
        raise RuntimeError("backend.chunk_ms must be an integer")
    if chunk_size_samples < 0:
        raise RuntimeError("backend.chunk_size_samples must be greater than or equal to zero")
    if chunk_ms < 0:
        raise RuntimeError("backend.chunk_ms must be greater than or equal to zero")
    if chunk_size_samples > 0 and chunk_ms > 0:
        raise RuntimeError("set only one of backend.chunk_size_samples or backend.chunk_ms")
    if chunk_size_samples > 0:
        return chunk_size_samples
    if chunk_ms > 0:
        samples = round(sample_rate_hz * chunk_ms / 1000.0)
        if samples <= 0:
            raise RuntimeError("backend.chunk_ms resolves to zero samples")
        return samples
    if emit_partial:
        return _DEFAULT_CHUNK_SIZE_SAMPLES
    return _DEFAULT_CHUNK_SIZE_SAMPLES


def _max_buffer_samples(*, sample_rate_hz: int, max_buffer_sec: float) -> int:
    if type(max_buffer_sec) is not float and type(max_buffer_sec) is not int:
        raise RuntimeError("backend.max_buffer_sec must be a number")
    if max_buffer_sec <= 0:
        raise RuntimeError("backend.max_buffer_sec must be positive")
    samples = round(sample_rate_hz * float(max_buffer_sec))
    if samples <= 0:
        raise RuntimeError("backend.max_buffer_sec resolves to zero samples")
    return samples


def _speech_energy_threshold(value: float) -> float:
    if type(value) is not float and type(value) is not int:
        raise RuntimeError("backend.speech_energy_threshold must be a number")
    threshold = float(value)
    if threshold < 0.0:
        raise RuntimeError("backend.speech_energy_threshold must be non-negative")
    return threshold


def _speech_energy_is_sufficient(samples: np.ndarray, threshold: float) -> bool:
    if samples.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
    return rms >= threshold


def _stream_result_from_text(
    text: str,
    *,
    sample_count: int,
    is_final: bool,
    allow_empty_text: bool,
) -> AsrStreamResult:
    transcript = build_asr_transcript(
        (
            AsrTranscriptSegment(
                start_sample=0,
                end_sample=sample_count,
                text=text.strip(),
            ),
        ),
        sample_count=sample_count,
        allow_empty_text=allow_empty_text,
    )
    return AsrStreamResult(
        transcript=transcript,
        is_final=is_final,
        sample_count=sample_count,
    )


def _validate_stream_request(request: AsrStreamRequest) -> None:
    if not request.session_id.strip():
        raise RuntimeError("ASR stream session_id is required")
    if type(request.user_turn_id) is not int:
        raise RuntimeError("ASR stream user_turn_id must be an integer")
    if request.user_turn_id <= 0:
        raise RuntimeError("ASR stream user_turn_id must be greater than zero")


def _transcription_text(result: str | _TranscriptionHypothesis) -> str:
    if isinstance(result, str):
        return result
    return result.text
