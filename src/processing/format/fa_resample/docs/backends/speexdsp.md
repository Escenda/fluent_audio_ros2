# speexdsp backend

`speexdsp` は realtime voice pipeline 向けの resampler backend である。
選択された場合は `libspeexdsp.so.1` を `dlopen` し、runtime dependency が無い場合は起動失敗にする。
`internal_linear_resampler` へ fallback しない。

## Role

- backend name: `speexdsp`
- runtime library: `libspeexdsp.so.1`
- processing path: `speex_resampler_process_interleaved_float`
- intended use: realtime voice frontend candidate
- input/output data type: interleaved float32

## Quality

`backend.quality` は必須で、integer `0..10` のみ許可する。
範囲外、未設定、integer 以外は起動失敗にする。

例:

```yaml
backend:
  name: speexdsp
  quality: 6
```

## Input Contract

- `stream_id` is non-empty
- positive input sample rate
- positive channel count
- `FLOAT32LE` encoding
- 32-bit sample depth
- `interleaved` layout
- non-empty data aligned to `channels * sizeof(float)`
- finite normalized samples in `[-1.0, 1.0]`

## Output Contract

- `encoding`: `FLOAT32LE`
- `bit_depth`: `32`
- `layout`: `interleaved`
- `channels`: input と同一
- `sample_rate`: `target_sample_rate`

SpeexDSP backend は sample format conversion、bit depth conversion、channel conversion、gain、clamp、denoise、VAD、ASR を行わない。

## Streaming State

The backend is stateful per `stream_id`.

The stream state stores:

- stream contract
- SpeexDSP resampler state
- cumulative input frame count
- cumulative output frame count
- SpeexDSP input latency
- SpeexDSP output latency

The stream contract includes input frame contract, target sample rate, backend name, and backend quality.
Changing those values mid-stream returns stream contract violation.

SpeexDSP can delay output. A process call with `output_frames == 0` can be a normal streaming result while latency is being absorbed. If the backend makes no input/output progress, it is backend process failure.

## Failure Policy

Startup failure:

- `target_sample_rate <= 0`
- `backend.quality` outside `0..10`
- `libspeexdsp.so.1` cannot be loaded
- required SpeexDSP symbols are missing

Frame rejection:

- unsupported frame contract
- empty `stream_id`
- non-finite or out-of-range sample
- mid-stream contract change

Runtime failure:

- SpeexDSP state creation fails for a stream
- `speex_resampler_process_interleaved_float` returns error
- process loop cannot make progress
- output samples cannot be encoded as finite normalized `FLOAT32LE`

## Metrics

The backend updates:

- `algorithmic_delay_input_samples`
- `algorithmic_delay_output_samples`
- `algorithmic_delay_ms`
- `processing_time_mean_ms`
- `processing_time_max_ms`
- `input_frames_total`
- `output_frames_total`
- `expected_output_frames`
- `frame_count_error_samples`

Algorithmic delay is taken from SpeexDSP latency APIs and exposed through diagnostics.

## Test Evidence

Automated tests cover selection, quality validation, no fallback to internal backend, simple smoke processing when runtime library is available, and optional quality metric comparison against SoXR `VHQ`.

検証済み報告では、base running container に SpeexDSP runtime は無く、smoke / metric validation のため `libspeexdsp1` が一時 install された。image-persistent dependency としての SpeexDSP は未検証である。
