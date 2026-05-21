from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

from fa_asr_py.backends.base import (
    ASR_AUDIO_ENCODING_PCM16LE,
    AsrBackendCapability,
    AsrRequest,
    AsrTranscript,
    AsrTranscriptSegment,
    build_asr_transcript,
)


_RIVA_LINEAR_PCM_ENCODING = "LINEAR_PCM"


@dataclass(frozen=True)
class RivaNimGrpcAsrConfig:
    server: str
    use_ssl: bool
    model: str
    language_code: str
    audio_encoding: str
    sample_rate_hz: int
    channels: int
    chunk_size_bytes: int
    interim_results: bool
    automatic_punctuation: bool
    enable_word_time_offsets: bool
    timeout_sec: float


@dataclass(frozen=True)
class RivaRecognitionConfig:
    encoding: str
    sample_rate_hz: int
    channels: int
    language_code: str
    model: str
    max_alternatives: int
    automatic_punctuation: bool
    enable_word_time_offsets: bool


@dataclass(frozen=True)
class RivaStreamingConfig:
    recognition_config: RivaRecognitionConfig
    interim_results: bool


@dataclass(frozen=True)
class RivaStreamingRequest:
    streaming_config: RivaStreamingConfig | None = None
    audio_content: bytes = b""


@dataclass(frozen=True)
class RivaWordInfo:
    start_time_sec: float
    end_time_sec: float
    word: str


@dataclass(frozen=True)
class RivaAlternative:
    transcript: str
    words: tuple[RivaWordInfo, ...] = ()


@dataclass(frozen=True)
class RivaStreamingResult:
    is_final: bool
    alternatives: tuple[RivaAlternative, ...]


@dataclass(frozen=True)
class RivaStreamingResponse:
    results: tuple[RivaStreamingResult, ...]


class RivaNimGrpcClient(Protocol):
    def streaming_recognize(
        self,
        requests: Iterable[RivaStreamingRequest],
        *,
        timeout_sec: float,
    ) -> Iterable[RivaStreamingResponse]:
        ...


class RivaNimGrpcClientFactory(Protocol):
    def create(self, config: RivaNimGrpcAsrConfig) -> RivaNimGrpcClient:
        ...


class NvidiaRivaNimGrpcClientFactory:
    def create(self, config: RivaNimGrpcAsrConfig) -> RivaNimGrpcClient:
        return NvidiaRivaNimGrpcClient(config)


class NvidiaRivaNimGrpcClient:
    def __init__(self, config: RivaNimGrpcAsrConfig) -> None:
        # nvidia-riva-client is required only when this backend is selected.
        # Importing here keeps other fa_asr backends testable in environments without Riva.
        import riva.client
        from riva.client.proto import riva_asr_pb2

        auth = riva.client.Auth(uri=config.server, use_ssl=config.use_ssl)
        self._service = riva.client.ASRService(auth)
        self._riva_asr_pb2 = riva_asr_pb2

    def streaming_recognize(
        self,
        requests: Iterable[RivaStreamingRequest],
        *,
        timeout_sec: float,
    ) -> Iterable[RivaStreamingResponse]:
        grpc_requests = self._grpc_requests(requests)
        grpc_responses = self._service.stub.StreamingRecognize(
            grpc_requests,
            timeout=timeout_sec,
        )
        for response in grpc_responses:
            yield self._internal_response(response)

    def _grpc_requests(self, requests: Iterable[RivaStreamingRequest]) -> Iterator:
        for request in requests:
            if request.streaming_config is not None:
                yield self._config_request(request.streaming_config)
                continue
            yield self._riva_asr_pb2.StreamingRecognizeRequest(
                audio_content=request.audio_content
            )

    def _config_request(self, streaming_config: RivaStreamingConfig):
        recognition = streaming_config.recognition_config
        config = self._riva_asr_pb2.RecognitionConfig(
            encoding=self._riva_asr_pb2.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=recognition.sample_rate_hz,
            language_code=recognition.language_code,
            max_alternatives=recognition.max_alternatives,
            enable_automatic_punctuation=recognition.automatic_punctuation,
            audio_channel_count=recognition.channels,
            enable_word_time_offsets=recognition.enable_word_time_offsets,
            model=recognition.model,
        )
        return self._riva_asr_pb2.StreamingRecognizeRequest(
            streaming_config=self._riva_asr_pb2.StreamingRecognitionConfig(
                config=config,
                interim_results=streaming_config.interim_results,
            )
        )

    @staticmethod
    def _internal_response(response) -> RivaStreamingResponse:
        results: list[RivaStreamingResult] = []
        for result in response.results:
            alternatives: list[RivaAlternative] = []
            for alternative in result.alternatives:
                alternatives.append(
                    RivaAlternative(
                        transcript=alternative.transcript,
                        words=tuple(
                            RivaWordInfo(
                                start_time_sec=float(word.start_time.seconds)
                                + float(word.start_time.nanos) / 1_000_000_000.0,
                                end_time_sec=float(word.end_time.seconds)
                                + float(word.end_time.nanos) / 1_000_000_000.0,
                                word=word.word,
                            )
                            for word in alternative.words
                        ),
                    )
                )
            results.append(
                RivaStreamingResult(
                    is_final=bool(result.is_final),
                    alternatives=tuple(alternatives),
                )
            )
        return RivaStreamingResponse(results=tuple(results))


class RivaNimGrpcAsrBackend:
    name = "riva_nim_grpc"

    def __init__(
        self,
        config: RivaNimGrpcAsrConfig,
        *,
        client_factory: RivaNimGrpcClientFactory | None = None,
    ) -> None:
        self._config = config
        self.capability = AsrBackendCapability(
            audio_encoding=ASR_AUDIO_ENCODING_PCM16LE,
            sample_rate_hz=config.sample_rate_hz,
            channels=config.channels,
            streaming=True,
            final_results_only=True,
        )
        factory = client_factory if client_factory is not None else NvidiaRivaNimGrpcClientFactory()
        self._client = factory.create(config)

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        request.payload.validate_matches(self.capability)
        responses = self._client.streaming_recognize(
            self._streaming_requests(request),
            timeout_sec=self._config.timeout_sec,
        )
        final_segments: list[AsrTranscriptSegment] = []
        for response in responses:
            final_segments.extend(self._final_segments(response, request=request))
        return build_asr_transcript(tuple(final_segments), sample_count=request.payload.sample_count)

    def _streaming_requests(self, request: AsrRequest) -> Iterator[RivaStreamingRequest]:
        yield RivaStreamingRequest(streaming_config=self._streaming_config())
        for start in range(0, len(request.payload.data), self._config.chunk_size_bytes):
            yield RivaStreamingRequest(
                audio_content=request.payload.data[start : start + self._config.chunk_size_bytes]
            )

    def _streaming_config(self) -> RivaStreamingConfig:
        return RivaStreamingConfig(
            recognition_config=RivaRecognitionConfig(
                encoding=_RIVA_LINEAR_PCM_ENCODING,
                sample_rate_hz=self._config.sample_rate_hz,
                channels=self._config.channels,
                language_code=self._config.language_code,
                model=self._config.model,
                max_alternatives=1,
                automatic_punctuation=self._config.automatic_punctuation,
                enable_word_time_offsets=self._config.enable_word_time_offsets,
            ),
            interim_results=self._config.interim_results,
        )

    def _final_segments(
        self,
        response: RivaStreamingResponse,
        *,
        request: AsrRequest,
    ) -> tuple[AsrTranscriptSegment, ...]:
        segments: list[AsrTranscriptSegment] = []
        for result in response.results:
            if not result.is_final:
                continue
            if not result.alternatives:
                continue
            alternative = result.alternatives[0]
            transcript = alternative.transcript.strip()
            if not transcript:
                continue
            segments.append(self._segment_from_alternative(alternative, request=request))
        return tuple(segments)

    def _segment_from_alternative(
        self,
        alternative: RivaAlternative,
        *,
        request: AsrRequest,
    ) -> AsrTranscriptSegment:
        transcript = alternative.transcript.strip()
        if not alternative.words:
            return AsrTranscriptSegment(
                start_sample=0,
                end_sample=request.payload.sample_count,
                text=transcript,
            )
        start_sec = min(word.start_time_sec for word in alternative.words)
        end_sec = max(word.end_time_sec for word in alternative.words)
        start_sample = self._seconds_to_floor_sample(start_sec)
        end_sample = self._seconds_to_ceil_sample(end_sec)
        return AsrTranscriptSegment(
            start_sample=max(0, start_sample),
            end_sample=min(request.payload.sample_count, end_sample),
            text=transcript,
        )

    def _seconds_to_floor_sample(self, seconds: float) -> int:
        if seconds < 0.0:
            raise RuntimeError("Riva word start_time must be non-negative")
        return int(seconds * float(self._config.sample_rate_hz))

    def _seconds_to_ceil_sample(self, seconds: float) -> int:
        if seconds <= 0.0:
            raise RuntimeError("Riva word end_time must be positive")
        sample = int(seconds * float(self._config.sample_rate_hz))
        if float(sample) / float(self._config.sample_rate_hz) < seconds:
            sample += 1
        return sample


def load_riva_nim_grpc_config(
    *,
    server: str,
    use_ssl: bool,
    model: str,
    language_code: str,
    audio_encoding: str,
    sample_rate_hz: int,
    channels: int,
    chunk_size_bytes: int,
    interim_results: bool,
    automatic_punctuation: bool,
    enable_word_time_offsets: bool,
    timeout_sec: float,
) -> RivaNimGrpcAsrConfig:
    normalized_server = server.strip()
    if not normalized_server:
        raise RuntimeError("backend.riva_nim_grpc.server is required")
    normalized_model = model.strip()
    if not normalized_model:
        raise RuntimeError("backend.model is required")
    normalized_language = language_code.strip()
    if not normalized_language:
        raise RuntimeError("backend.language is required")
    if audio_encoding.strip() != ASR_AUDIO_ENCODING_PCM16LE:
        raise RuntimeError("backend.riva_nim_grpc.audio_encoding must be PCM16LE")
    if sample_rate_hz <= 0:
        raise RuntimeError("backend.riva_nim_grpc.sample_rate_hz must be greater than zero")
    if channels != 1:
        raise RuntimeError("backend.riva_nim_grpc.channels must be 1")
    if chunk_size_bytes <= 0:
        raise RuntimeError("backend.riva_nim_grpc.chunk_size_bytes must be greater than zero")
    if chunk_size_bytes % 2 != 0:
        raise RuntimeError("backend.riva_nim_grpc.chunk_size_bytes must align to PCM16 samples")
    if timeout_sec <= 0.0:
        raise RuntimeError("backend.timeout_sec must be greater than zero")
    return RivaNimGrpcAsrConfig(
        server=normalized_server,
        use_ssl=use_ssl,
        model=normalized_model,
        language_code=normalized_language,
        audio_encoding=ASR_AUDIO_ENCODING_PCM16LE,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        chunk_size_bytes=chunk_size_bytes,
        interim_results=interim_results,
        automatic_punctuation=automatic_punctuation,
        enable_word_time_offsets=enable_word_time_offsets,
        timeout_sec=timeout_sec,
    )
