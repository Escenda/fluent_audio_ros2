# internal_linear_fade

## 1. 概要

`internal_linear_fade` は `fa_fade` ノードから呼び出される ROS2 非依存 C++ backend である。外部 process、device、file、network、モデル推論 backend は使わない。

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

同じ sample frame 内の全 channel は同じ position の gain を共有する。出力 byte列と `position_frames_` は、全 sample の検証と変換が成功した場合だけ commit する。

## 4. 安全境界

この backend は値の修正を行わない。不正入力または不正出力を検出した場合、呼び出し元は frame を drop する。
`position_frames_ + frame_count` が `uint64_t` を overflow する場合も失敗として返す。未知の `ProcessStatus` / `FadeMode` は `std::logic_error` で fail closed する。

## 5. 非責務

gain parameter、limiter、normalize、filter、denoise、resampling、sample format conversion、channel conversion、device I/O は含めない。
