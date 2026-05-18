# internal_stft backend

## 目的

`internal_stft` は ROS2 に依存しない deterministic STFT backend である。AI model runtime ではなく、`numpy` による signal feature extraction を行う。

## Runtime

- Python package: `numpy`
- ROS2 import: なし
- external process: なし
- model path: なし

## Input

`numpy.ndarray` の 1D `float32` sample array。sample は finite かつ normalized `[-1.0, 1.0]` でなければならない。

## Output

`StftResult`:

- `frame_count`
- `bin_count`
- `real`: shape `(frame_count, bin_count)` / `float32`
- `imag`: shape `(frame_count, bin_count)` / `float32`

## Failure Policy

config 不正は backend construction 時に例外にする。input contract 不一致は `ValueError`、output non-finite は `RuntimeError` とする。padding、clip、resample、dtype conversion は行わない。
