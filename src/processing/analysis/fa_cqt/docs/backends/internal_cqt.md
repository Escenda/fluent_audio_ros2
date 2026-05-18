# internal_cqt backend

`internal_cqt` は NumPy 実装の deterministic complex CQT backend である。

## 入力

- `numpy.ndarray`
- dtype: `float32`
- shape: one-dimensional
- value range: normalized `[-1.0, 1.0]`

## 出力

- `frame_count`
- `real`: shape `frame_count x bin_count`、dtype `float32`
- `imag`: shape `frame_count x bin_count`、dtype `float32`
- `center_frequencies_hz`: shape `bin_count`、dtype `float32`
- `window_lengths`: shape `bin_count`、dtype `uint32`

## 境界

この backend は ROS2 topic、ROS message、parameter server、device I/O を知らない。format conversion、resample、downmix、normalization、padding、truncate は実行しない。
