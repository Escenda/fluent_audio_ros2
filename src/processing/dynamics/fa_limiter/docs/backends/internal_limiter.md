# internal_limiter

`fa_limiter` は外部 model runtime や cloud API を持たない。backend は node 内部の deterministic FLOAT32LE hard limiter 処理である。

## Contract

- 入力と出力は `FLOAT32LE` / 32bit / interleaved。
- `threshold.linear` は有限値かつ `(0.0, 1.0]`。
- sample rate、bit depth、channel count、layout は変更しない。
- sample が `threshold.linear` を超える場合は `threshold.linear` に制限する。
- sample が `-threshold.linear` を下回る場合は `-threshold.linear` に制限する。

## Dependencies

- ROS2 `rclcpp`
- `fa_interfaces/msg/AudioFrame`
- `diagnostic_msgs`

外部 DSP library、Python runtime、model file は不要。
