# internal_gain

`internal_gain` は `fa_gain` package 内の ROS 非依存 deterministic backend である。
外部 model runtime や cloud API を持たない。ROS node は parameter、AudioFrame metadata
検証、diagnostics、publish/subscribe のみを持つ。

## Contract

- 入力と出力は validated `FLOAT32LE` / 32bit / interleaved sample bytes。
- `gain.linear` は有限値かつ `>= 0.0`。
- limiter、normalize は行わない。
- 入力または出力が `[-1.0, 1.0]` 範囲外になる frame は `ProcessStatus` で拒否する。
- 失敗した frame では output buffer を commit しない。

## Dependencies

- C++ standard library のみ。
- 外部 DSP library、Python runtime、model file は不要。
- `rclcpp`、`fa_interfaces/msg/AudioFrame`、`diagnostic_msgs` には依存しない。
