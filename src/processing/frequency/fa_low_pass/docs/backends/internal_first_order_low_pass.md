# internal_first_order_low_pass

`internal_first_order_low_pass` は `fa_low_pass_node` に内蔵された一次 low-pass backend である。

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

## Failure Policy

- 不正 config: 起動失敗。
- 不正 frame: warning を出して drop。
- 不正 sample: frame 全体を drop し、state は更新しない。
