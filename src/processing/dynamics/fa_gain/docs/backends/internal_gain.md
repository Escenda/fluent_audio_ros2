# internal_gain

`fa_gain` は外部 model runtime や cloud API を持たない。backend は node 内部の deterministic FLOAT32LE gain 処理である。

## Contract

- 入力と出力は `FLOAT32LE` / 32bit / interleaved。
- `gain.linear` は有限値かつ `>= 0.0`。
- clipping、limiter、normalize は行わない。
- 範囲外になる frame は drop する。

## Dependencies

- ROS2 `rclcpp`
- `fa_interfaces/msg/AudioFrame`
- `diagnostic_msgs`

外部 DSP library、Python runtime、model file は不要。
