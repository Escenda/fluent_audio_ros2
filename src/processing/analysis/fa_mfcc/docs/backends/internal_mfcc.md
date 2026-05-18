# internal_mfcc backend

`internal_mfcc` は NumPy 実装の deterministic MFCC backend である。

## 入力

- `numpy.ndarray`
- dtype: `float32`
- shape: one-dimensional
- value range: normalized `[-1.0, 1.0]`

## 出力

- `frame_count`
- `values`: shape `frame_count x n_mfcc`、dtype `float32`

## 境界

この backend は ROS2 topic、ROS message、parameter server、device I/O を知らない。format conversion、resample、downmix、normalization、padding、truncate は実行しない。
