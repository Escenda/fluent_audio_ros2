# internal_threshold_gate

## 1. Backend

`internal_threshold_gate` は `fa_noise_gate_node.cpp` 内で完結する組み込み backend である。
外部 DSP library、device backend、resampler、format converter は使用しない。

## 2. 入出力

- input: `FLOAT32LE` interleaved samples in normalized range `[-1.0, 1.0]`
- output: `FLOAT32LE` interleaved samples in normalized range `[-1.0, 1.0]`

## 3. Contract

backend は invalid input を補正しない。node の frame validation と sample validation により、
不正 frame は publish されない。

## 4. Algorithm

```text
y = x * closed_gain_linear  when abs(x) < threshold_linear
y = x                       otherwise
```

この backend は compressor、limiter、normalize、filter、denoise ではない。
