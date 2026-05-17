# internal_log_mel backend

`internal_log_mel` は ROS-free な deterministic feature extraction backend である。

## Input

- `numpy.ndarray`
- dtype: `float32`
- shape: 1 次元
- value range: `[-1.0, 1.0]`

## Output

- `frame_count x n_mels` の `float32` matrix
- natural-log mel power

## Failure Policy

不正な config、non-finite input、normalized range 外 input、`feature.n_fft` 未満の input は例外にする。zero padding、clamp、resample、downmix、model fallback は行わない。
