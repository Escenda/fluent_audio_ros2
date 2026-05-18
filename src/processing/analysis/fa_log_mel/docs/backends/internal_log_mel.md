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

## Formula

`internal_log_mel` は `numpy.hanning` window、unnormalized `numpy.fft.rfft` power、HTK mel scale、area-normalized triangular mel filterbank、natural log を使う。zero signal の出力は全要素が `log(feature.log_floor)` になる。

## Failure Policy

不正な config、non-finite input、normalized range 外 input、`feature.n_fft` 未満の input、`feature.n_fft + N * feature.hop_length` に一致しない input は例外にする。zero padding、truncate、clamp、resample、downmix、model fallback は行わない。
