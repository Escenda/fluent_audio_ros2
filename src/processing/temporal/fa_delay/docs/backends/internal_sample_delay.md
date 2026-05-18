# internal_sample_delay

## 1. 概要

`internal_sample_delay` は `fa_delay` ノードから呼ばれる ROS 非依存 C++ backend である。外部 process、device、file、network、モデル推論 backend は使わない。

## 2. 入力

- FLOAT32LE
- bit depth 32
- interleaved
- sample range `[-1.0, 1.0]`
- `channels > 0`
- `data.size()` は `channels * sizeof(float)` の倍数

## 3. 処理

`delay.ms` を `expected.sample_rate` に基づき whole samples に変換し、channel ごとの delay line を作る。delay line は configured silence `0.0F` で初期化する。

```text
output[channel] = delay_line[channel].front()
delay_line[channel].push_back(input[channel])
```

## 4. 安全境界

この backend は値の修正を行わない。不正入力を検出した場合、`ProcessStatus` を返し、delay buffer と output buffer を更新しない。

## 5. 非責務

gain parameter、limiter、normalize、filter、denoise、resampling、sample format conversion、channel conversion、device I/O、echo、reverb は含めない。
