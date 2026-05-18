# internal_static_curve backend

## 1. Backend

`internal_static_curve` は `fa_compressor` package 内の ROS 非依存 deterministic backend
である。外部 DSP library、device API、resampler、limiter、gate、normalize backend は使わない。
ROS node は parameter、AudioFrame metadata 検証、diagnostics、publish/subscribe のみを持つ。

## 2. 入力

- validated `FLOAT32LE` interleaved sample bytes
- finite normalized sample
- config: `channels`, `threshold_linear`, `ratio`, `makeup_gain_linear`

## 3. 処理

sample ごとに `threshold + (abs(sample) - threshold) / ratio` の knee なし静的圧縮を行い、符号を戻して makeup gain を掛ける。

## 4. 出力

backend は `ProcessResult{status, samples_compressed}` を返す。出力 sample が正規化範囲を
超える場合は `kOutOfRangeOutput` を返し、output buffer と compressed sample counter を
commit しない。node は frame を publish せず drop counter で可視化する。

backend は `rclcpp`、`fa_interfaces/msg/AudioFrame`、`diagnostic_msgs` に依存しない。
