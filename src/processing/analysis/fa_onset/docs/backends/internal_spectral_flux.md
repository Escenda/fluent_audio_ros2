# internal_spectral_flux backend

`internal_spectral_flux` は NumPy 実装の deterministic onset measurement backend である。

## 入力

- `numpy.ndarray`
- dtype: `float32`
- shape: one-dimensional
- value range: normalized `[-1.0, 1.0]`

## 出力

- `frame_count`
- `frame_times_sec`: shape `frame_count`、dtype `float32`
- `strengths`: shape `frame_count`、dtype `float32`
- `detected`: shape `frame_count`、dtype `bool`

## 境界

この backend は ROS2 topic、ROS message、parameter server、device I/O を知らない。format conversion、resample、downmix、normalization、padding、truncate は実行しない。
