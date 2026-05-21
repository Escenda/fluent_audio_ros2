# speexdsp backend

`speexdsp` は realtime voice pipeline 向けの resampler backend である。
選択された場合は `libspeexdsp.so.1` を `dlopen` し、runtime dependency が無い場合は起動失敗にする。
`internal_linear_resampler` へ fallback しない。

## Role

- backend name: `speexdsp`
- runtime library: `libspeexdsp.so.1`
- package contract: `package.xml` declares `libspeexdsp1` as `exec_depend`
- processing path: `speex_resampler_process_interleaved_float`
- intended use: realtime voice frontend candidate
- input/output data type: interleaved float32

The C++ backend loads `libspeexdsp.so.1` at runtime. It does not include SpeexDSP headers or link against a SpeexDSP dev package at build time.

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

`test/cpp/test_resample_graph.cpp` includes a SpeexDSP ROS graph smoke with `backend.name=speexdsp` and
`backend.quality=6`. The smoke starts `FaResampleNode`, publishes and subscribes `AudioFrame`, and checks
`/diagnostics` for `backend.name`, `backend.quality`, `algorithmic_delay_ms`, `processing_time_mean_ms`,
`input_frames_total`, `output_frames_total`, `expected_output_frames`, and `frame_count_error_samples`.
This verifies backend selection on the ROS graph. Real-device smoke evidence is tracked separately below.

When `libspeexdsp.so.1` is available, the quality comparison test records `speex_q6_passband_*`
GTest XML properties for both test-only quality metrics and backend metrics:

- `rms_error` / `peak_error` / `snr_db` / `compared_samples`
- `algorithmic_delay_input_samples` / `algorithmic_delay_output_samples` / `algorithmic_delay_ms`
- `processing_time_mean_ms` / `processing_time_max_ms`
- `input_frames_total` / `output_frames_total` / `frame_count_error_samples`

`lib/fa_resample/fa_resample_metrics_report.py` is installed as a stdlib-only metrics reporter. It reads
`fa_resample_backend_test.gtest.xml` or a directory of `fa_resample` GTest XML files, renders the quality and
backend metrics above, and fails incomplete metric groups instead of silently omitting missing evidence.

Verified metrics report values include:

| metric group | rms_error | peak_error | snr_db | algorithmic_delay_ms |
| --- | --- | --- | --- | --- |
| `speex_q6_passband` | `5.941855108704546e-07` | `1.370906829833984e-06` | `1.094336419058643e+02` | `3.000000000000000e+00` |

Interpretation:

SpeexDSP q6 metrics are comparisons against SoXR `VHQ` reference for the test signal. They are test-only
waveform fidelity evidence, not direct human MOS or direct ASR accuracy evidence.

In the verified passband report, SpeexDSP q6 has `snr_db=1.094336419058643e+02`,
`rms_error=5.941855108704546e-07`, and `peak_error=1.370906829833984e-06`, with
`algorithmic_delay_ms=3.000000000000000e+00`. Compared with SoXR HQ in the same report, SpeexDSP q6 is about
26.36 dB lower in SNR, so its error power is roughly hundreds of times larger and its RMS error is roughly
tens of times larger for this test signal. The tradeoff is much lower algorithmic delay.

Use SpeexDSP q6 as a low-latency realtime voice pipeline candidate when the added waveform error is acceptable
for the downstream VAD / ASR / dialogue path being evaluated. Do not read this report as proof that SpeexDSP q6
is worse or better for every human-perception or ASR scenario; it only proves the measured closeness to the
chosen SoXR VHQ reference under the tested signals.

## Real-device Smoke

SpeexDSP quality `6` has been verified in the current running VLAbor container with PowerConf S3 through a
`fluent_audio_system` temp config:

```text
fa_in -> fa_sample_format -> fa_resample
```

Verified output:

- output topic: `/fa_real_speexdsp/audio/resample16k`
- output stream: `audio/real_speexdsp/preprocessed/mono16k`
- source: `hw:CARD=S3,DEV=0`
- sample rate: `16000`
- encoding: `FLOAT32LE`

Verified diagnostics included:

- `backend.name=speexdsp`
- `backend.quality=6`
- `algorithmic_delay_ms=3.000000`
- `processing_time_mean_ms=0.066197`
- `input_frames_total=74880`
- `output_frames_total=24960`
- `frame_count_error_samples=0`

This proves selected-backend real-device smoke in the current running container. It does not prove release or
publish adoption of the parent-repo overlay image.

## VLAbor Overlay Image Evidence

`ros2_ws/src/vlabor_ros2/docker/Dockerfile.local` が `libsoxr0` / `libspeexdsp1` を持つとは扱わない。
heavy VLAbor base image の full rebuild も `fa_resample` completion proof ではない。

親 repo の `docker/vlabor/Dockerfile.fluent-audio` と `docker/vlabor/build-fluent-audio-image` は、
`ghcr.io/takatronix/vlabor-local:latest` から薄い `vlabor-fluent-audio:local` overlay image を作り、
`libsoxr0` / `libspeexdsp1` だけを追加する contract である。

検証済み報告:

- `docker/vlabor/build-fluent-audio-image` は約 12 秒で成功した。
- base digest は `sha256:b709...`、出力 image は `vlabor-fluent-audio:local`。
- apt 出力では `libsoxr0 is already the newest version` と `libspeexdsp1` の新規 install を確認した。
- `docker run --rm --entrypoint bash vlabor-fluent-audio:local -lc 'dpkg-query -W libsoxr0 libspeexdsp1 && ls -l /lib/x86_64-linux-gnu/libsoxr.so.0* /lib/x86_64-linux-gnu/libspeexdsp.so.1*'`
  は成功し、`libsoxr0:amd64 0.1.3-4build2`、`libspeexdsp1:amd64 1.2~rc1.2-1.1ubuntu3`、
  および両方の `.so` symlink / file を確認した。
- overlay image の検証は build 成功と `docker run` による `dpkg-query` / `.so` dependency check までである。
- `fa_resample` package build/test は current running VLAbor container へ
  `docker compose -f docker/vlabor/compose.yml exec vlabor` で入って実行され、
  `51 tests, 0 errors, 0 failures, 0 skipped` で通過している。

overlay files は親 repo の未コミット編集である。compose / deployment を `vlabor-fluent-audio:local` に
切り替えた状態での package test、release / publish 採用は未検証であり、別途運用判断が必要である。
