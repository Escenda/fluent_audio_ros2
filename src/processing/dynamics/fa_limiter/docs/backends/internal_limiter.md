# internal_limiter

`internal_limiter` は `fa_limiter` package 内の ROS 非依存 deterministic backend である。
外部 model runtime や cloud API を持たない。ROS node は parameter、AudioFrame metadata
検証、diagnostics、publish/subscribe のみを持つ。

## Contract

- 入力と出力は validated `FLOAT32LE` / 32bit / interleaved sample bytes。
- `threshold.linear` は有限値かつ `(0.0, 1.0]`。
- sample rate、bit depth、channel count、layout は変更しない。
- sample が `threshold.linear` を超える場合は `threshold.linear` に制限する。
- sample が `-threshold.linear` を下回る場合は `-threshold.linear` に制限する。
- finite sample を threshold に丸めることは limiter の明示処理であり、fallback ではない。
- non-finite sample、empty input、frame byte alignment error は `ProcessStatus` で拒否する。
- 失敗した frame では output buffer と limited sample counter を commit しない。

## Dependencies

- C++ standard library のみ。
- 外部 DSP library、Python runtime、model file は不要。
- `rclcpp`、`fa_interfaces/msg/AudioFrame`、`diagnostic_msgs` には依存しない。
