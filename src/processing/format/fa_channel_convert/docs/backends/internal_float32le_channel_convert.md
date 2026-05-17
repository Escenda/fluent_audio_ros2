# internal_float32le_channel_convert

## Backend

この package の backend は C++ node 内の internal FLOAT32LE channel converter である。外部 DSP backend、ML model、device driver、resampler は使用しない。

## Contract

- 入力: `AudioFrame.data` as FLOAT32LE little-endian bytes
- layout: interleaved
- mode: `mono_to_stereo_duplicate` または `stereo_to_mono_average`
- output: FLOAT32LE little-endian bytes

## Failure handling

設定不正は startup fail closed とする。runtime frame 不正は warning を出して frame を drop し、diagnostics counter に反映する。
