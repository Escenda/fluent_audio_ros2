# internal_overlap_add backend

## 目的

`internal_overlap_add` は、ROS2 node 内で完結する overlap-add accumulator である。外部 process、device、model runtime は使わない。

## Input format

- `fa_interfaces/msg/AudioFrame`
- `FLOAT32LE`
- `bit_depth=32`
- `layout=interleaved`
- `data.size() == window.frame_samples * channels * sizeof(float)`
- sample 値は finite かつ `[-1.0, 1.0]`

## Output format

- `fa_interfaces/msg/AudioFrame`
- `data.size() == window.hop_samples * channels * sizeof(float)`
- `source_id` は入力を保持
- `stream_id` は `output_topic`
- `epoch` は node lifetime 内で単調増加

## Failure policy

- invalid input frame は drop し、accumulator を変更しない
- source / format / future epoch gap は accumulator reset
- duplicate / regressing input epoch は stale audio replay を避けるため drop
- non-finite output、range 外 output、buffer overflow は reset/drop
- clamp、tail padding、zero fill は行わない

## Diagnostics

`diagnostics` の `backend` value は `internal_overlap_add` とする。config、input/output/drop counters、epoch regression drop count、accumulated chunk count、reset count、buffered sample frame count を出す。

## Test fixture

小さな synthetic FLOAT32LE chunk を使う。大きな録音 fixture は不要。
