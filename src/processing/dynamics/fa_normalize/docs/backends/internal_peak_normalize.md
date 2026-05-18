# internal_peak_normalize backend

`internal_peak_normalize` は ROS 非依存の C++ peak normalization backend である。ROS2 topic、`fa_interfaces/msg/AudioFrame`、parameter、diagnostics、publisher/subscriber は知らない。外部 process、device、codec、model には依存しない。

## 入力

- `std::vector<uint8_t>` の `FLOAT32LE` sample bytes
- `channels * sizeof(float)` に整列した interleaved frame
- normalized sample range `[-1.0, 1.0]`

## 処理

1. input bytes が空でなく、frame boundary に整列していることを確認する。
2. 全 sample を `float` として読み、finite と normalized range を確認する。
3. `peak = max(abs(sample))` を計算する。
4. `peak < silence_threshold_linear` なら output bytes を input bytes と同一にし、`ProcessMode::kSilencePassthrough` と `gain = 1.0` を返す。
5. それ以外は `gain = target_peak_linear / peak` を全 sample に適用し、`ProcessMode::kNormalized` と gain を返す。
6. 出力 sample が finite かつ normalized range 内であることを確認する。

## 禁止事項

- clamp
- limiter
- compressor
- gate
- filter
- denoise
- loudness / LUFS normalization
- sample format conversion
- channel conversion
- resampling
- device I/O

拒否時は `ProcessStatus` を返し、output vector は更新しない。warning、drop counter、`last_gain` の commit は ROS node 側の責務である。
