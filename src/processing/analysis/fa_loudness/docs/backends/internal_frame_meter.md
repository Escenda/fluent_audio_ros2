# internal_frame_meter backend

## 目的

`internal_frame_meter` は ROS2 に依存しない deterministic frame meter backend である。AI model runtime ではなく、`numpy` と標準 `math` による RMS / peak / dBFS measurement を行う。

## Runtime

- Python package: `numpy`
- ROS2 import: なし
- external process: なし
- model path: なし

## Input

`numpy.ndarray` の 1D `float32` sample array。sample は finite かつ normalized `[-1.0, 1.0]` でなければならない。

## Output

`LoudnessResult`:

- `sample_count`
- `rms`
- `peak`
- `rms_dbfs`
- `peak_dbfs`
- `crest_factor`

## Failure Policy

config 不正は backend construction 時に例外にする。input contract 不一致は `ValueError`、output non-finite は `RuntimeError` とする。padding、clip、resample、dtype conversion、gain correction は行わない。
