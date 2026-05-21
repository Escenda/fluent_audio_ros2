import struct
from typing import Iterable

import numpy as np
import pytest

from fa_asr_py.backends.base import AsrAudioPayload, AsrRequest, asr_transcript_text
from fa_asr_py.backends.riva_nim_grpc import (
    RivaAlternative,
    RivaNimGrpcAsrBackend,
    RivaNimGrpcAsrConfig,
    RivaNimGrpcClient,
    RivaNimGrpcClientFactory,
    RivaStreamingRequest,
    RivaStreamingResponse,
    RivaStreamingResult,
    RivaWordInfo,
    load_riva_nim_grpc_config,
)


class _FakeRivaClient:
    def __init__(self, responses: tuple[RivaStreamingResponse, ...]) -> None:
        self.responses = responses
        self.requests: tuple[RivaStreamingRequest, ...] = ()
        self.timeout_sec = 0.0

    def streaming_recognize(
        self,
        requests: Iterable[RivaStreamingRequest],
        *,
        timeout_sec: float,
    ) -> tuple[RivaStreamingResponse, ...]:
        self.requests = tuple(requests)
        self.timeout_sec = timeout_sec
        return self.responses


class _FakeRivaClientFactory:
    def __init__(self, client: _FakeRivaClient) -> None:
        self.client = client

    def create(self, config: RivaNimGrpcAsrConfig) -> RivaNimGrpcClient:
        del config
        return self.client


def _config() -> RivaNimGrpcAsrConfig:
    return load_riva_nim_grpc_config(
        server="riva.example.test:50051",
        use_ssl=True,
        model="parakeet-rnnt-multilingual",
        language_code="ja-JP",
        audio_encoding="PCM16LE",
        sample_rate_hz=16000,
        channels=1,
        chunk_size_bytes=4,
        interim_results=False,
        automatic_punctuation=True,
        enable_word_time_offsets=True,
        timeout_sec=5.0,
    )


def _pcm16_request() -> AsrRequest:
    data = struct.pack("<" + ("h" * 64), *range(100, 164))
    return AsrRequest(
        session_id="session-1",
        user_turn_id=3,
        payload=AsrAudioPayload.from_pcm16le_bytes(
            data,
            sample_rate_hz=16000,
            channels=1,
        ),
    )


def _response_with_words() -> RivaStreamingResponse:
    return RivaStreamingResponse(
        results=(
            RivaStreamingResult(
                is_final=True,
                alternatives=(
                    RivaAlternative(
                        transcript="こんにちは 世界",
                        words=(
                            RivaWordInfo(
                                start_time_sec=0.001,
                                end_time_sec=0.002,
                                word="こんにちは",
                            ),
                            RivaWordInfo(
                                start_time_sec=0.002,
                                end_time_sec=0.004,
                                word="世界",
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def test_riva_nim_grpc_sends_config_first_then_original_pcm16_chunks() -> None:
    request = _pcm16_request()
    client = _FakeRivaClient((_response_with_words(),))
    backend = RivaNimGrpcAsrBackend(
        _config(),
        client_factory=_FakeRivaClientFactory(client),
    )

    transcript = backend.transcribe(request)

    assert asr_transcript_text(transcript) == "こんにちは 世界"
    assert client.timeout_sec == 5.0
    expected_chunks = tuple(
        request.payload.data[start : start + 4] for start in range(0, len(request.payload.data), 4)
    )
    assert len(client.requests) == 1 + len(expected_chunks)
    first = client.requests[0]
    assert first.streaming_config is not None
    assert first.audio_content == b""
    recognition = first.streaming_config.recognition_config
    assert recognition.encoding == "LINEAR_PCM"
    assert recognition.sample_rate_hz == 16000
    assert recognition.channels == 1
    assert recognition.model == "parakeet-rnnt-multilingual"
    audio_requests = client.requests[1:]
    assert all(request.streaming_config is None for request in audio_requests)
    assert tuple(request.audio_content for request in audio_requests) == expected_chunks
    assert b"".join(request.audio_content for request in audio_requests) == request.payload.data


def test_riva_nim_grpc_maps_word_timings_to_sample_offsets() -> None:
    client = _FakeRivaClient((_response_with_words(),))
    backend = RivaNimGrpcAsrBackend(
        _config(),
        client_factory=_FakeRivaClientFactory(client),
    )

    transcript = backend.transcribe(_pcm16_request())

    assert transcript.segments[0].start_sample == 16
    assert transcript.segments[0].end_sample == 64


def test_riva_nim_grpc_rejects_float32_without_conversion() -> None:
    client = _FakeRivaClient((_response_with_words(),))
    backend = RivaNimGrpcAsrBackend(
        _config(),
        client_factory=_FakeRivaClientFactory(client),
    )
    request = AsrRequest(
        session_id="session-1",
        user_turn_id=3,
        payload=AsrAudioPayload.from_float32_samples(
            np.zeros(160, dtype=np.float32),
            sample_rate_hz=16000,
        ),
    )

    with pytest.raises(ValueError, match="encoding must be PCM16LE"):
        backend.transcribe(request)
    assert client.requests == ()


def test_riva_nim_grpc_fails_when_no_final_transcript() -> None:
    client = _FakeRivaClient(
        (
            RivaStreamingResponse(
                results=(
                    RivaStreamingResult(
                        is_final=False,
                        alternatives=(RivaAlternative(transcript="partial"),),
                    ),
                )
            ),
        )
    )
    backend = RivaNimGrpcAsrBackend(
        _config(),
        client_factory=_FakeRivaClientFactory(client),
    )

    with pytest.raises(RuntimeError, match="segments must not be empty"):
        backend.transcribe(_pcm16_request())


def test_riva_nim_grpc_config_rejects_non_pcm16_input_contract() -> None:
    with pytest.raises(RuntimeError, match="audio_encoding must be PCM16LE"):
        load_riva_nim_grpc_config(
            server="riva.example.test:50051",
            use_ssl=False,
            model="parakeet-rnnt-multilingual",
            language_code="ja-JP",
            audio_encoding="FLOAT32LE",
            sample_rate_hz=16000,
            channels=1,
            chunk_size_bytes=3200,
            interim_results=False,
            automatic_punctuation=True,
            enable_word_time_offsets=True,
            timeout_sec=5.0,
        )
