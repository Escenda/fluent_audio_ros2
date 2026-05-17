# internal_linear_fade

## 1. 概要

`internal_linear_fade` は `fa_fade` ノード内で実行する C++ 実装 backend である。外部 process、device、file、network、モデル推論 backend は使わない。

## 2. 入力

- FLOAT32LE
- bit depth 32
- interleaved
- sample range `[-1.0, 1.0]`
- `channels > 0`
- `data.size()` は `channels * sizeof(float)` の倍数

## 3. 処理

フレーム位置 `position` と `duration_frames` から `double` gain を計算し、各 sample に乗算する。

```text
fade_in:  min(1, position / duration)
fade_out: max(0, 1 - position / duration)
```

## 4. 安全境界

この backend は値の修正を行わない。不正入力または不正出力を検出した場合、呼び出し元は frame を drop する。

## 5. 非責務

gain parameter、limiter、normalize、filter、denoise、resampling、sample format conversion、channel conversion、device I/O は含めない。
