from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fa_asr_py.backends.base import (
    ASR_AUDIO_ENCODING_FLOAT32LE,
    AsrAudioPayload,
    AsrRequest,
    AsrStreamRequest,
    asr_transcript_text,
)
import fa_asr_py.backends.parakeet_multilingual_buffered as parakeet_backend_module
from fa_asr_py.backends.factory import AsrBackendSettings, build_asr_backend
from fa_asr_py.backends.parakeet_multilingual_buffered import (
    ParakeetMultilingualBufferedAsrBackend,
    ParakeetMultilingualBufferedConfig,
    load_parakeet_multilingual_buffered_config,
)


class _FakeParakeetRunner:
    instances: list["_FakeParakeetRunner"] = []
    scripted_results: list[str] = []

    def __init__(self, config: ParakeetMultilingualBufferedConfig) -> None:
        self.config = config
        self.transcribe_calls: list[np.ndarray] = []
        _FakeParakeetRunner.instances.append(self)

    def transcribe(self, samples: np.ndarray) -> str:
        self.transcribe_calls.append(samples.copy())
        if _FakeParakeetRunner.scripted_results:
            return _FakeParakeetRunner.scripted_results.pop(0)
        return "こんにちは 世界"


def _config(
    *,
    emit_partial: bool = True,
    chunk_size_samples: int = 2,
    speech_energy_threshold: float = 0.001,
) -> ParakeetMultilingualBufferedConfig:
    return load_parakeet_multilingual_buffered_config(
        model="nvidia/parakeet-multilingual-1.1b",
        model_path_value="",
        language="",
        language_policy="auto_detect",
        sample_rate_hz=16000,
        channels=1,
        chunk_size_samples=chunk_size_samples,
        chunk_ms=0,
        emit_partial=emit_partial,
        max_buffer_sec=30.0,
        speech_energy_threshold=speech_energy_threshold,
    )


def _payload(samples: np.ndarray, *, sample_rate_hz: int = 16000) -> AsrAudioPayload:
    return AsrAudioPayload.from_float32_samples(
        samples,
        sample_rate_hz=sample_rate_hz,
        channels=1,
    )


def test_config_rejects_model_that_does_not_identify_multilingual_parakeet_1_1b(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="multilingual Parakeet 1.1B"):
        load_parakeet_multilingual_buffered_config(
            model="nvidia/parakeet-ctc-0.6b",
            model_path_value="",
            language="",
            language_policy="auto_detect",
            sample_rate_hz=16000,
            channels=1,
            chunk_ms=0,
            emit_partial=False,
        )

    with pytest.raises(RuntimeError, match="multilingual Parakeet 1.1B"):
        load_parakeet_multilingual_buffered_config(
            model="nvidia/general-asr-model",
            model_path_value="",
            language="",
            language_policy="auto_detect",
            sample_rate_hz=16000,
            channels=1,
            chunk_ms=0,
            emit_partial=False,
        )

    model_path = tmp_path / "parakeet-1.1b-multilingual.nemo"
    model_path.write_bytes(b"model")

    config = load_parakeet_multilingual_buffered_config(
        model="",
        model_path_value=str(model_path),
        language="",
        language_policy="auto_detect",
        sample_rate_hz=16000,
        channels=1,
        chunk_size_samples=1600,
        chunk_ms=0,
        emit_partial=True,
    )

    assert config.model == ""
    assert config.model_path == model_path.resolve()


def test_config_fails_closed_on_unsupported_audio_contract() -> None:
    with pytest.raises(RuntimeError, match="backend.sample_rate_hz must be 16000"):
        load_parakeet_multilingual_buffered_config(
            model="nvidia/parakeet-multilingual-1.1b",
            model_path_value="",
            language="",
            language_policy="auto_detect",
            sample_rate_hz=8000,
            channels=1,
            chunk_ms=0,
            emit_partial=False,
        )

    with pytest.raises(RuntimeError, match="backend.channels must be 1"):
        load_parakeet_multilingual_buffered_config(
            model="nvidia/parakeet-multilingual-1.1b",
            model_path_value="",
            language="",
            language_policy="auto_detect",
            sample_rate_hz=16000,
            channels=2,
            chunk_ms=0,
            emit_partial=False,
        )

    with pytest.raises(RuntimeError, match="backend.language_policy must be auto_detect"):
        load_parakeet_multilingual_buffered_config(
            model="nvidia/parakeet-multilingual-1.1b",
            model_path_value="",
            language="",
            language_policy="ja",
            sample_rate_hz=16000,
            channels=1,
            chunk_ms=0,
            emit_partial=True,
        )

    with pytest.raises(RuntimeError, match="backend.language is not supported"):
        load_parakeet_multilingual_buffered_config(
            model="nvidia/parakeet-multilingual-1.1b",
            model_path_value="",
            language="ja",
            language_policy="auto_detect",
            sample_rate_hz=16000,
            channels=1,
            chunk_ms=0,
            emit_partial=True,
        )


def test_factory_builds_buffered_backend_without_worker_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeParakeetRunner.instances.clear()
    monkeypatch.setattr(
        parakeet_backend_module,
        "_ParakeetMultilingualBufferedRunner",
        _FakeParakeetRunner,
    )

    backend = build_asr_backend(
        AsrBackendSettings(
            name="parakeet_multilingual_buffered",
            model="nvidia/parakeet-multilingual-1.1b",
            language="",
            language_policy="auto_detect",
            sample_rate_hz=16000,
            channels=1,
            chunk_size_samples=1600,
            emit_partial=True,
        )
    )

    assert isinstance(backend, ParakeetMultilingualBufferedAsrBackend)
    assert len(_FakeParakeetRunner.instances) == 1
    assert backend.capability.audio_encoding == ASR_AUDIO_ENCODING_FLOAT32LE
    assert backend.capability.sample_rate_hz == 16000
    assert backend.capability.channels == 1
    assert backend.capability.streaming is True
    assert backend.capability.final_results_only is False


def test_transcribe_accepts_only_float32le_16khz_mono() -> None:
    backend = ParakeetMultilingualBufferedAsrBackend(
        _config(),
        runner_class=_FakeParakeetRunner,
    )
    request = AsrRequest(
        session_id="s1",
        user_turn_id=1,
        payload=_payload(np.array([0.1, -0.1, 0.0, 0.2], dtype=np.float32)),
    )

    transcript = backend.transcribe(request)

    assert asr_transcript_text(transcript) == "こんにちは 世界"

    with pytest.raises(ValueError, match="encoding must be FLOAT32LE"):
        backend.transcribe(
            AsrRequest(
                session_id="s1",
                user_turn_id=1,
                payload=AsrAudioPayload.from_pcm16le_bytes(
                    b"\x00\x00\x00\x00",
                    sample_rate_hz=16000,
                    channels=1,
                ),
            )
        )

    with pytest.raises(ValueError, match="sample_rate_hz must be 16000"):
        backend.transcribe(
            AsrRequest(
                session_id="s1",
                user_turn_id=1,
                payload=_payload(
                    np.array([0.1, 0.2], dtype=np.float32),
                    sample_rate_hz=8000,
                ),
            )
        )

    with pytest.raises(ValueError, match="channels must be 1"):
        backend.transcribe(
            AsrRequest(
                session_id="s1",
                user_turn_id=1,
                payload=AsrAudioPayload.from_float32_samples(
                    np.array([0.1, 0.2], dtype=np.float32),
                    sample_rate_hz=16000,
                    channels=2,
                ),
            )
        )


def test_streaming_redecodes_rolling_buffer_for_partial_and_final() -> None:
    _FakeParakeetRunner.instances.clear()
    _FakeParakeetRunner.scripted_results = [
        "こんにちは",
        "こんにちは 世界",
        "こんにちは 世界",
    ]
    backend = ParakeetMultilingualBufferedAsrBackend(
        _config(),
        runner_class=_FakeParakeetRunner,
    )
    session = backend.start_stream(AsrStreamRequest(session_id="s1", user_turn_id=5))

    first = np.array([0.1, 0.2], dtype=np.float32)
    second = np.array([-0.1, 0.0, 0.3], dtype=np.float32)
    first_partial = session.push_audio(_payload(first))
    assert session.drain_results() == ()
    second_partial = session.push_audio(_payload(second))

    results = session.finish()

    assert len(first_partial) == 1
    assert first_partial[0].is_final is False
    assert asr_transcript_text(first_partial[0].transcript) == "こんにちは"
    assert len(second_partial) == 1
    assert second_partial[0].is_final is False
    assert asr_transcript_text(second_partial[0].transcript) == "こんにちは 世界"
    assert len(results) == 1
    assert results[0].is_final is True
    assert results[0].sample_count == 5
    assert asr_transcript_text(results[0].transcript) == "こんにちは 世界"
    assert len(_FakeParakeetRunner.instances[0].transcribe_calls) == 3
    np.testing.assert_array_equal(
        _FakeParakeetRunner.instances[0].transcribe_calls[-1],
        np.concatenate((first, second)),
    )

    with pytest.raises(RuntimeError, match="already finished"):
        session.drain_results()


def test_streaming_decodes_only_retained_rolling_buffer_window() -> None:
    _FakeParakeetRunner.instances.clear()
    _FakeParakeetRunner.scripted_results = [
        "a",
        "b",
        "c",
        "final",
    ]
    config = replace(_config(), max_buffer_samples=4)
    backend = ParakeetMultilingualBufferedAsrBackend(
        config,
        runner_class=_FakeParakeetRunner,
    )
    session = backend.start_stream(AsrStreamRequest(session_id="s1", user_turn_id=5))

    first = np.array([0.10, 0.20], dtype=np.float32)
    second = np.array([0.30, 0.40], dtype=np.float32)
    third = np.array([0.50, 0.60], dtype=np.float32)
    session.push_audio(_payload(first))
    session.push_audio(_payload(second))
    session.push_audio(_payload(third))
    final = session.finish()

    assert asr_transcript_text(final[0].transcript) == "final"
    np.testing.assert_array_equal(
        _FakeParakeetRunner.instances[0].transcribe_calls[-1],
        np.concatenate((second, third)),
    )


def test_streaming_finish_and_cancel_fail_closed() -> None:
    backend = ParakeetMultilingualBufferedAsrBackend(
        _config(),
        runner_class=_FakeParakeetRunner,
    )
    empty_session = backend.start_stream(AsrStreamRequest(session_id="empty", user_turn_id=1))
    with pytest.raises(RuntimeError, match="requires buffered audio"):
        empty_session.finish()

    cancelled = backend.start_stream(AsrStreamRequest(session_id="cancelled", user_turn_id=1))
    cancelled.push_audio(_payload(np.array([0.1], dtype=np.float32)))
    cancelled.cancel()
    with pytest.raises(RuntimeError, match="already cancelled"):
        cancelled.finish()


def test_streaming_rejects_empty_final_when_speech_energy_is_sufficient() -> None:
    _FakeParakeetRunner.instances.clear()
    _FakeParakeetRunner.scripted_results = [""]
    backend = ParakeetMultilingualBufferedAsrBackend(
        _config(emit_partial=False, speech_energy_threshold=0.05),
        runner_class=_FakeParakeetRunner,
    )
    session = backend.start_stream(AsrStreamRequest(session_id="speech", user_turn_id=1))
    session.push_audio(_payload(np.array([0.3, -0.3, 0.25, -0.25], dtype=np.float32)))

    with pytest.raises(RuntimeError, match="final transcript must not be empty"):
        session.finish()


def test_streaming_allows_empty_final_only_when_speech_energy_is_below_threshold() -> None:
    _FakeParakeetRunner.instances.clear()
    _FakeParakeetRunner.scripted_results = [""]
    backend = ParakeetMultilingualBufferedAsrBackend(
        _config(emit_partial=False, speech_energy_threshold=0.05),
        runner_class=_FakeParakeetRunner,
    )
    session = backend.start_stream(AsrStreamRequest(session_id="silence", user_turn_id=1))
    session.push_audio(_payload(np.array([0.001, -0.001, 0.0, 0.001], dtype=np.float32)))

    results = session.finish()

    assert len(results) == 1
    assert results[0].is_final is True
    assert results[0].sample_count == 4
    assert asr_transcript_text(results[0].transcript) == ""
