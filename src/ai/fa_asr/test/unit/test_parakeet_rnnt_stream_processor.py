from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from fa_asr_py.backends.parakeet_rnnt_stream_processor import (
    ContextSize,
    DecodeStepResult,
    EncodedAudio,
    ParakeetRnntStreamProcessor,
    ParakeetStreamConfig,
)


@dataclass
class FakeEncoded:
    frames: list[str]

    def __getitem__(self, key: object) -> "FakeEncoded":
        batch_slice, time_slice = key
        assert batch_slice == slice(None, None, None)
        assert isinstance(time_slice, slice)
        start = 0 if time_slice.start is None else time_slice.start
        return FakeEncoded(self.frames[start:])


@dataclass
class FakeBatchContext:
    left: int
    chunk: np.ndarray


class FakeBuffer:
    def __init__(self, context_samples: ContextSize) -> None:
        self.context_size = context_samples
        self.context_size_batch = context_samples
        self.samples = "buffered-samples"
        self.add_calls: list[dict[str, Any]] = []

    def add_audio_batch(
        self,
        samples: np.ndarray,
        *,
        audio_lengths: Any,
        is_last_chunk: bool,
        is_last_chunk_batch: Any,
    ) -> None:
        self.add_calls.append(
            {
                "samples": samples.copy(),
                "audio_lengths": audio_lengths,
                "is_last_chunk": is_last_chunk,
                "is_last_chunk_batch": is_last_chunk_batch,
            }
        )


class FakeHyps:
    def __init__(self, parts: list[str]) -> None:
        self.parts = parts

    def merge_(self, other: "FakeHyps") -> None:
        self.parts.extend(other.parts)


class FakeRuntime:
    sample_rate = 16000
    feature_stride_sec = 0.01
    encoder_subsampling_factor = 4
    uses_chunked_limited_attention_with_right_context = True

    def __init__(self) -> None:
        self.configure_calls = 0
        self.attention_context: ContextSize | None = None
        self.buffer: FakeBuffer | None = None
        self.decode_calls: list[dict[str, Any]] = []
        self.trim_calls: list[int] = []
        self.encode_calls = 0

    def configure_for_streaming(self) -> None:
        self.configure_calls += 1

    def set_default_attention_context(self, context_encoder_frames: ContextSize) -> None:
        self.attention_context = context_encoder_frames

    def create_buffer(self, context_samples: ContextSize) -> FakeBuffer:
        self.buffer = FakeBuffer(context_samples)
        return self.buffer

    def make_audio_lengths(self, sample_count: int) -> int:
        return sample_count

    def make_last_chunk_batch(self, is_last_chunk: bool) -> bool:
        return is_last_chunk

    def encode(self, buffer: FakeBuffer) -> EncodedAudio:
        self.encode_calls += 1
        encoded = FakeEncoded([f"left-{self.encode_calls}", f"chunk-{self.encode_calls}"])
        return EncodedAudio(output=encoded, output_lengths=np.array([9]))

    def subsample_context(self, context: ContextSize, *, factor: int) -> FakeBatchContext:
        return FakeBatchContext(
            left=context.left // factor,
            chunk=np.array([context.chunk // factor]),
        )

    def trim_left_context(self, encoded_output: FakeEncoded, *, left_frames: int) -> FakeEncoded:
        self.trim_calls.append(left_frames)
        return encoded_output[:, left_frames:]

    def output_length(
        self,
        encoded: EncodedAudio,
        context_batch: FakeBatchContext,
        *,
        is_last_chunk: bool,
    ) -> Any:
        if is_last_chunk:
            return encoded.output_lengths - context_batch.left
        return context_batch.chunk

    def decode(
        self,
        encoded_output: FakeEncoded,
        output_lengths: Any,
        *,
        previous_state: Any,
    ) -> DecodeStepResult:
        index = len(self.decode_calls) + 1
        self.decode_calls.append(
            {
                "encoded_output": encoded_output,
                "output_lengths": output_lengths,
                "previous_state": previous_state,
            }
        )
        return DecodeStepResult(
            chunk_hypotheses=FakeHyps([f"word{index}"]),
            decoder_state=f"state{index}",
        )

    def merge_hypotheses(self, current_hypotheses: Any, chunk_hypotheses: FakeHyps) -> FakeHyps:
        if current_hypotheses is None:
            return chunk_hypotheses
        current_hypotheses.merge_(chunk_hypotheses)
        return current_hypotheses

    def hypotheses_to_text(self, hypotheses: FakeHyps) -> str:
        return " ".join(hypotheses.parts)


def _config() -> ParakeetStreamConfig:
    return ParakeetStreamConfig(
        model_path="/fake/model.nemo",
        sample_rate=16000,
        left_context_secs=0.04,
        chunk_secs=0.08,
        right_context_secs=0.04,
    )


def test_processor_derives_contexts_and_configures_runtime_attention() -> None:
    runtime = FakeRuntime()
    processor = ParakeetRnntStreamProcessor(_config(), runtime=runtime)

    assert runtime.configure_calls == 1
    assert processor.context_encoder_frames == ContextSize(left=1, chunk=2, right=1)
    assert processor.context_samples == ContextSize(left=640, chunk=1280, right=640)
    assert processor.encoder_frame2audio_samples == 640
    assert runtime.attention_context == ContextSize(left=1, chunk=2, right=1)


def test_push_decodes_first_chunk_plus_right_then_chunk_and_preserves_decoder_state() -> None:
    runtime = FakeRuntime()
    processor = ParakeetRnntStreamProcessor(_config(), runtime=runtime)

    assert processor.push(np.zeros(1919, dtype=np.float32)) == []

    first = processor.push(np.zeros(1, dtype=np.float32))
    assert len(first) == 1
    assert first[0].text == "word1"
    assert first[0].accepted_samples == 1920
    assert first[0].complete is False
    assert runtime.buffer is not None
    assert runtime.buffer.add_calls[0]["samples"].size == 1920
    assert runtime.buffer.add_calls[0]["is_last_chunk"] is False
    assert runtime.decode_calls[0]["previous_state"] is None

    second = processor.push(np.zeros(1280, dtype=np.float32))
    assert len(second) == 1
    assert second[0].text == "word1 word2"
    assert second[0].accepted_samples == 3200
    assert runtime.buffer.add_calls[1]["samples"].size == 1280
    assert runtime.decode_calls[1]["previous_state"] == "state1"


def test_finish_flushes_pending_tail_as_last_chunk_and_returns_final_snapshot() -> None:
    runtime = FakeRuntime()
    processor = ParakeetRnntStreamProcessor(_config(), runtime=runtime)

    processor.push(np.zeros(1920, dtype=np.float32))
    processor.push(np.zeros(100, dtype=np.float32))
    final = processor.finish()

    assert final.text == "word1 word2"
    assert final.accepted_samples == 2020
    assert final.complete is True
    assert runtime.buffer is not None
    assert runtime.buffer.add_calls[-1]["samples"].size == 100
    assert runtime.buffer.add_calls[-1]["is_last_chunk"] is True
    assert runtime.decode_calls[-1]["previous_state"] == "state1"
    np.testing.assert_array_equal(runtime.decode_calls[-1]["output_lengths"], np.array([8]))


def test_reset_clears_stream_state_and_creates_new_buffer() -> None:
    runtime = FakeRuntime()
    processor = ParakeetRnntStreamProcessor(_config(), runtime=runtime)
    original_buffer = processor.buffer

    processor.push(np.zeros(1920, dtype=np.float32))
    processor.reset()

    assert processor.pending.size == 0
    assert processor.decoder_state is None
    assert processor.current_hypotheses is None
    assert processor.accepted_samples == 0
    assert processor.decode_steps == 0
    assert processor.buffer is not original_buffer


def test_rejects_invalid_audio_payloads() -> None:
    processor = ParakeetRnntStreamProcessor(_config(), runtime=FakeRuntime())

    with pytest.raises(TypeError, match="numpy.ndarray"):
        processor.push([0.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mono 1-D"):
        processor.push(np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        processor.push(np.array([np.nan], dtype=np.float32))

    assert processor.push(np.array([2.0], dtype=np.float64)) == []
    assert processor.pending.dtype == np.float32
    assert processor.pending[0] == 1.0
