# internal_linear_resampler backend

`internal_linear_resampler` は ROS 非依存の `FLOAT32LE` interleaved linear resampling backend である。
この backend は debug / reference 用の内部 backend であり、missing `speexdsp` / `soxr` runtime の fallback ではない。

## Role

- backend name: `internal_linear_resampler`
- quality label: `debug_reference`
- runtime dependency: なし
- intended use: debug、reference、minimal environment smoke
- production quality role: high-quality resampling の根拠にはしない

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

The backend emits `FLOAT32LE` / 32-bit / interleaved bytes at the configured `target_sample_rate`.
The target sample rate must be greater than zero.
The output channel count is identical to the input channel count.

`ProcessResult` returns explicit status, frame contract status, and output frame count.
Invalid input is not represented by an empty output vector alone.

## Streaming State

The backend is stateful per `stream_id`.

The stream state stores:

- stream contract
- cumulative input frame count
- cumulative output frame count
- next output frame index
- retained input tail buffer
- retained buffer start frame index

The stream contract includes input frame contract, target sample rate, backend name, and backend quality.
Changing those values mid-stream returns stream contract violation.

The linear interpolation uses cumulative output frame index, so chunk boundaries do not reset the interpolation phase. The previous tail frame is retained for interpolation across chunk boundaries.

## Failure Policy

Invalid frames return typed `ProcessStatus` and leave the caller-owned output vector unchanged.
The ROS node is responsible for warning logs, drop counters, and publish suppression.

The backend does not decode PCM16/PCM32, does not encode PCM16, does not change channels, and does not clamp overflow samples. Those operations belong to explicit processing nodes in the system pipeline.

## Metrics

The backend updates:

- `processing_time_mean_ms`
- `processing_time_max_ms`
- `input_frames_total`
- `output_frames_total`
- `expected_output_frames`
- `frame_count_error_samples`

Algorithmic delay metrics remain zero for this backend because it has no library-reported filter delay.
