# internal_first_order_low_pass

`internal_first_order_low_pass` は ROS2 非依存の C++ backend `fa_low_pass::backends::InternalFirstOrderLowPassBackend` である。

## 入力

- `FLOAT32LE`
- `bit_depth == 32`
- `layout == interleaved`
- finite normalized samples in `[-1.0, 1.0]`

## 出力

入力と同じ frame metadata を保持し、`stream_id` と `data` のみを更新する。出力 sample も
finite normalized range に収まる必要がある。

## 設計境界

この backend は low-pass recurrence のみを担当する。device I/O、sample format
conversion、resampling、gain、normalize、limiter、denoise は別 package の責務である。
ROS2 topic、ROS message、`rclcpp` は参照しない。

source 切り替えの判断は node 側が行い、backend は `reset_state` 指示を受けた frame で一時 state を未初期化にして処理する。frame 全体が成功した場合だけ reset 後 state を commit する。

## Failure Policy

- 不正 config: 起動失敗。
- 空入力は `kEmptyInput` として拒否する。
- `channels * sizeof(float)` に揃わない byte 長は `kMisalignedInput` として拒否する。
- non-finite input sample は `kNonFiniteInput` として拒否する。
- normalized range 外の input sample は `kOutOfRangeInput` として拒否する。
- non-finite output sample は `kNonFiniteOutput` として拒否する。
- normalized range 外の output sample は `kOutOfRangeOutput` として拒否する。
- 拒否時は channel filter state を更新しない。
