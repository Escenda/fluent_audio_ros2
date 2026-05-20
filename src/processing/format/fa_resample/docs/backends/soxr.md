# soxr backend

`soxr` は high-quality resampling / recording / evaluation 向けの resampler backend である。
選択された場合は `libsoxr.so.0` を `dlopen` し、runtime dependency が無い場合は起動失敗にする。
`internal_linear_resampler` へ fallback しない。

## Role

- backend name: `soxr`
- runtime library: `libsoxr.so.0`
- processing path: SoXR streaming API
- intended use: high-quality resampling, recording, evaluation, golden reference
- input/output data type: interleaved float32

## Quality

`backend.quality` は必須で、次の値だけを許可する。

- `QQ`
- `LQ`
- `MQ`
- `HQ`
- `VHQ`

例:

```yaml
backend:
  name: soxr
  quality: MQ
```

`VHQ` は quality metric test の golden reference として使う。realtime default として採用するかは別途 latency / CPU cost の実測で判断する。

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

SoXR backend は sample format conversion、bit depth conversion、channel conversion、gain、clamp、denoise、VAD、ASR を行わない。

## Streaming State

The backend is stateful per `stream_id`.

The stream state stores:

- stream contract
- SoXR state
- cumulative input frame count
- cumulative output frame count
- `soxr_delay` output delay

The stream contract includes input frame contract, target sample rate, backend name, and backend quality.
Changing those values mid-stream returns stream contract violation.

The backend uses SoXR streaming process calls and does not recreate SoXR state for every chunk.
SoXR can delay output. A process call with `output_frames == 0` can be a normal streaming result while filter delay is being absorbed. If the backend makes no input/output progress, it is backend process failure.

## Failure Policy

Startup failure:

- `target_sample_rate <= 0`
- `backend.quality` outside `QQ` / `LQ` / `MQ` / `HQ` / `VHQ`
- `libsoxr.so.0` cannot be loaded
- required SoXR symbols are missing

Frame rejection:

- unsupported frame contract
- empty `stream_id`
- non-finite or out-of-range sample
- mid-stream contract change

Runtime failure:

- SoXR state creation fails for a stream
- SoXR process returns error
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

Algorithmic delay is derived from `soxr_delay`: output sample delay is exposed directly, input sample delay is converted by `input_rate / output_rate`, and ms delay is converted by output sample rate.

## Required Test Path

Automated tests must cover:

- backend selection and no fallback to internal backend
- quality parsing for `QQ` / `LQ` / `MQ` / `HQ` / `VHQ`
- simple runtime smoke when `libsoxr.so.0` is available
- passband multi-tone comparison against SoXR `VHQ`
- out-of-band alias probe comparison against SoXR `VHQ`
- impulse peak offset measurement
- diagnostics delay / frame count metrics

検証済み報告では、VLAbor image に SoXR runtime が存在し、Docker/VLAbor container 内の `fa_resample` package build/test が通過している。
