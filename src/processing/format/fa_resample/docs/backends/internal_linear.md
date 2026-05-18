# internal_linear_resampler backend

`internal_linear_resampler` は ROS 非依存の `FLOAT32LE` interleaved linear resampling backend である。ROS2 topic、`fa_interfaces/msg/AudioFrame`、publisher/subscriber、diagnostics は知らない。

## Input Contract

- positive input sample rate
- positive channel count
- `FLOAT32LE` encoding
- 32-bit sample depth
- `interleaved` layout
- non-empty data aligned to `channels * sizeof(float)`
- finite normalized samples in `[-1.0, 1.0]`

## Output Contract

The backend emits `FLOAT32LE` / 32-bit / interleaved bytes at the configured `target_sample_rate`. The target sample rate must be greater than zero.

`ProcessResult` returns explicit status, frame contract status, and output frame count. Invalid input is not represented by an empty output vector alone.

## Failure Policy

Invalid frames return typed `ProcessStatus` and leave the caller-owned output vector unchanged. The ROS node is responsible for warning logs, drop counters, and publish suppression.

The backend does not decode PCM16/PCM32, does not encode PCM16, and does not clamp overflow samples. Those operations belong to explicit format/dynamics nodes in the system pipeline.

If a higher quality resampler is introduced later, it must be represented as an explicit backend or package contract.
