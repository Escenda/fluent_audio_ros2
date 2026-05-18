# internal_crossfade backend

`internal_crossfade` は explicit overlap と fade curve に基づいて segment 接続を行う backend contract である。

## Required Config

- `overlap.frames`
- `fade.curve`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.bit_depth`
- `expected.layout`

## Forbidden

- boundary guessing
- hidden limiter / compressor
- hidden resample
- hidden channel conversion
- missing segment 補完
- ROS2 topic/message dependency inside backend

