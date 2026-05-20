# soxr backend

`soxr` は high-quality resampling / recording / evaluation 向けの resampler backend である。
選択された場合は `libsoxr.so.0` を `dlopen` し、runtime dependency が無い場合は起動失敗にする。
`internal_linear_resampler` へ fallback しない。

## Role

- backend name: `soxr`
- runtime library: `libsoxr.so.0`
- package contract: `package.xml` declares `libsoxr0` as `exec_depend`
- processing path: SoXR streaming API
- intended use: high-quality resampling, recording, evaluation, golden reference
- input/output data type: interleaved float32

The C++ backend loads `libsoxr.so.0` at runtime. It does not include SoXR headers or link against a SoXR dev package at build time.

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

`test/cpp/test_resample_graph.cpp` includes a SoXR ROS graph smoke with `backend.name=soxr` and
`backend.quality=MQ`. The smoke starts `FaResampleNode`, publishes and subscribes `AudioFrame`, and checks
`/diagnostics` for `backend.name`, `backend.quality`, `algorithmic_delay_ms`, `processing_time_mean_ms`,
`input_frames_total`, `output_frames_total`, `expected_output_frames`, and `frame_count_error_samples`.
This verifies backend selection on the ROS graph. Real-device smoke evidence is tracked separately below.

The SoXR graph smoke once failed with `FLOAT32LE encoding failed` when it used a high-amplitude ramp.
The production input contract remains finite normalized `FLOAT32LE`; only the graph smoke input was changed
to a low-amplitude multi-tone signal.

The quality comparison test records backend metrics in GTest XML for SoXR prefixes such as
`passband_soxr_vhq_reference`, `passband_soxr_mq`, `passband_soxr_hq`, `alias_soxr_vhq_reference`,
`alias_soxr_mq`, and `alias_soxr_hq`.

The XML separates test-only quality metrics from backend delay / processing / frame metrics:

- quality: `rms_error` / `peak_error` / `snr_db` / `compared_samples`
- backend metrics: `algorithmic_delay_input_samples` / `algorithmic_delay_output_samples` / `algorithmic_delay_ms`
- backend metrics: `processing_time_mean_ms` / `processing_time_max_ms`
- backend metrics: `input_frames_total` / `output_frames_total` / `frame_count_error_samples`

`lib/fa_resample/fa_resample_metrics_report.py` is installed as a stdlib-only metrics reporter. It reads
`fa_resample_backend_test.gtest.xml` or a directory of `fa_resample` GTest XML files, renders the quality and
backend metrics above, and fails incomplete metric groups instead of silently omitting missing evidence.

## Real-device Smoke

SoXR/MQ has been verified in the current running VLAbor container with PowerConf S3 through a
`fluent_audio_system` temp config:

```text
fa_in -> fa_sample_format -> fa_resample
```

Verified output:

- output topic: `/fa_real_soxr/audio/resample16k`
- output stream: `audio/real_soxr/preprocessed/mono16k`
- source: `hw:CARD=S3,DEV=0`
- sample rate: `16000`
- encoding: `FLOAT32LE`

Verified diagnostics included:

- `backend.name=soxr`
- `backend.quality=MQ`
- `algorithmic_delay_ms=22.500000`
- `processing_time_mean_ms=0.056465`
- `input_frames_total=72000`
- `output_frames_total=23640`
- `frame_count_error_samples=-360`

This proves selected-backend real-device smoke in the current running container.

検証済み報告では、running container 内で `libsoxr0` が見えており、Docker/VLAbor container 内の
`fa_resample` package build/test は `51 tests, 0 errors, 0 failures, 0 skipped` で通過している。
fresh VLAbor image check
`docker run --rm --entrypoint bash ghcr.io/takatronix/vlabor-local:latest` では、
`libsoxr0:amd64 0.1.3-4build2` と `/lib/x86_64-linux-gnu/libsoxr.so.0` が確認済みである。
この証跡は current fresh VLAbor image に SoXR runtime が存在することを示すが、Dockerfile /
entrypoint policy の修正や、親 repo / `vlabor_ros2` の変更を主張するものではない。
