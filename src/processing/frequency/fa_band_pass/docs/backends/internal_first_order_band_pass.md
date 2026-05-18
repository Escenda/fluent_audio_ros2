# internal_first_order_band_pass backend

## Role

`internal_first_order_band_pass` は ROS2 非依存の C++ backend `fa_band_pass::backends::InternalFirstOrderBandPassBackend` である。外部 process、device、file、network I/O を持たない。

## Input

FLOAT32LE interleaved sample bytes。node 境界で `AudioFrame` contract validation 済みの frame の `data` だけを受け取り、ROS2 topic、ROS message、`rclcpp` は参照しない。

## Processing

1. accepted source の初回 sample で channel state を初期化する。
2. 各 channel に一次 high-pass recurrence を適用する。
3. high-pass output を同じ channel の一次 low-pass recurrence に入力する。
4. final output sample を FLOAT32LE interleaved `data` へ書き戻す。

## Failure Policy

backend は意味を変える fallback を行わない。

- 空入力は `kEmptyInput` として拒否する。
- `channels * sizeof(float)` に揃わない byte 長は `kMisalignedInput` として拒否する。
- non-finite input sample は `kNonFiniteInput` として拒否する。
- normalized range 外の input sample は `kOutOfRangeInput` として拒否する。
- FLOAT32LE として表現できない intermediate / final output は `kNonFiniteOutput` として拒否する。
- normalized range 外の final output は `kOutOfRangeOutput` として拒否する。
- 拒否時は channel filter state を更新しない。

source 切り替えの判断は node 側が行い、backend は `reset_state` 指示を受けた frame で一時 state を未初期化にして処理する。frame 全体が成功した場合だけ reset 後 state を commit する。
