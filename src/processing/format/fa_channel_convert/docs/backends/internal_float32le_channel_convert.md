# internal_float32le_channel_convert

## Backend

この package の backend は `include/fa_channel_convert/backends/internal_float32le_channel_convert.hpp` と `src/backends/internal_float32le_channel_convert.cpp` に置く ROS 非依存の internal FLOAT32LE channel converter である。外部 DSP backend、ML model、device driver、resampler は使用しない。

ROS node は parameter、topic、diagnostics、drop counter だけを扱い、channel-count conversion と frame format contract は backend に委譲する。

## Contract

- 入力: `AudioFrame.data` as FLOAT32LE little-endian bytes
- layout: interleaved
- mode: `mono_to_stereo_duplicate` または `stereo_to_mono_average`
- output: FLOAT32LE little-endian bytes
- backend input: ROS message ではなく `FrameContract` と byte buffer
- backend output: `ProcessResult` と caller-owned output buffer

## Failure handling

設定不正は backend constructor で startup fail closed とする。runtime frame 不正は `FrameContractStatus` / `ProcessStatus` として返し、output buffer は更新しない。ROS node はその status を warning にし、frame を drop して diagnostics counter に反映する。

empty vector を失敗 sentinel として扱わない。失敗理由は必ず typed status に載せる。
