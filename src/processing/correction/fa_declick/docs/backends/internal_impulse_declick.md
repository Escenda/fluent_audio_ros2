# internal_impulse_declick

## 1. 概要

`internal_impulse_declick` は `fa_declick` ノード内で実行する C++ 実装 backend である。外部 process、device、file、network、モデル推論 backend は使わない。

## 2. 入力

- FLOAT32LE
- bit depth 32
- interleaved
- sample range `[-1.0, 1.0]`
- `channels > 0`
- `data.size()` は `channels * sizeof(float)` の倍数

## 3. 処理

同一 channel の previous / current / next を比較し、current が両隣から `threshold.delta` より大きく離れ、両隣同士が `threshold.delta` 以内で近い場合に current を click とみなす。

```text
output[current] = (previous + next) / 2
```

`window.max_samples` が `2` 以上の場合は、最大 `window.max_samples` までの連続 impulse run に同じ境界条件を適用する。

## 4. 安全境界

この backend はサンプルの明示補正だけを行う。不正入力を検出した場合、呼び出し元は frame を drop する。正規化範囲外の出力を clamp して publish することはない。

## 5. 非責務

gain parameter、limiter、normalize、filter、denoise、declip、decrackle、resampling、sample format conversion、channel conversion、device I/O、echo、reverb は含めない。
