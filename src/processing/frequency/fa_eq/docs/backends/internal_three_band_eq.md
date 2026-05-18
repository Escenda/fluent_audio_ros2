# internal_three_band_eq backend

## Role

`internal_three_band_eq` は ROS2 非依存の C++ backend `fa_eq::backends::InternalThreeBandEqBackend` である。外部 process、device、file、network I/O を持たない。

## Input

FLOAT32LE interleaved sample bytes。node 境界で `AudioFrame` contract validation 済みの frame の `data` だけを受け取り、ROS2 topic、ROS message、`rclcpp` は参照しない。

## Processing

1. accepted source の初回 sample で channel state を初期化する。
2. 各 channel に一次 low-pass recurrence を適用して low band を得る。
3. 同じ input sample に一次 high-pass recurrence を適用して high band を得る。
4. `input - low - high` を mid band とする。
5. `low * gain_low + mid * gain_mid + high * gain_high` を final output sample として FLOAT32LE interleaved `data` へ書き戻す。

## Failure Policy

backend は意味を変える fallback を行わない。

- 空入力は `kEmptyInput` として拒否する。
- `channels * sizeof(float)` に揃わない byte 長は `kMisalignedInput` として拒否する。
- non-finite input sample は `kNonFiniteInput` として拒否する。
- normalized range 外の input sample は `kOutOfRangeInput` として拒否する。
- FLOAT32LE として表現できない split / mix output は `kNonFiniteOutput` として拒否する。
- normalized range 外の final output は `kOutOfRangeOutput` として拒否する。
- 拒否時は channel filter state を更新しない。

source 切り替えの判断は node 側が行い、backend は `reset_state` 指示を受けた frame で一時 state を未初期化にして処理する。frame 全体が成功した場合だけ reset 後 state を commit する。
