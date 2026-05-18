# internal_crossfade backend

`internal_crossfade` は explicit overlap と fade curve に基づいて2つの FLOAT32LE segment を接続する ROS-free C++ backend である。

backend は ROS2 topic、`fa_interfaces/msg/AudioFrame`、diagnostics、publisher/subscriber を知らない。

## Required Config

- `expected.channels`
- `crossfade.overlap_frames`
- `crossfade.curve`

## Input

- `segment_a`: FLOAT32LE interleaved sample bytes
- `segment_b`: FLOAT32LE interleaved sample bytes
- both byte lengths are multiples of `channels * sizeof(float)`
- both segments have at least `overlap_frames`
- all samples are finite normalized FLOAT32 in `[-1.0, 1.0]`

## Output

backend writes one output byte vector only after all validation succeeds.

```text
output = a_prefix + crossfade(a_tail, b_head) + b_suffix
```

## Forbidden

- boundary guessing
- hidden limiter / compressor
- hidden resample
- hidden channel conversion
- missing segment 補完
- ROS2 topic/message dependency inside backend

unknown `FadeCurve` / `ProcessStatus` は `std::logic_error` で fail closed する。
