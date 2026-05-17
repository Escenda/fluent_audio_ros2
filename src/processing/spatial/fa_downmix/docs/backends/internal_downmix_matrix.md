# internal_downmix_matrix

## Backend

この package の backend は C++ node 内の internal downmix matrix である。外部 DSP backend、ML model、device driver、resampler は使用しない。

## Contract

- 入力: `AudioFrame.data` as FLOAT32LE little-endian bytes
- layout: interleaved
- mode: `average_to_mono` または `pair_average_to_stereo`
- output: FLOAT32LE little-endian bytes

## Matrix

`average_to_mono` は全入力 channel に `1 / input_channels` の係数を持つ 1 行 matrix と等価である。

`pair_average_to_stereo` は偶数 index channel を left、奇数 index channel を right に割り当て、各 pair 群に `1 / (input_channels / 2)` の係数を持つ 2 行 matrix と等価である。

## Failure handling

設定不正は startup fail closed とする。runtime frame 不正は warning を出して frame を drop し、diagnostics counter に反映する。
