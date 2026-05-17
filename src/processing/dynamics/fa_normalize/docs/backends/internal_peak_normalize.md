# internal_peak_normalize backend

`internal_peak_normalize` は `fa_normalize_node` 内部で完結する C++ peak normalization backend である。外部 process、device、codec、model には依存しない。

## 入力

- `fa_interfaces/msg/AudioFrame`
- `FLOAT32LE`
- `bit_depth == 32`
- `layout == interleaved`
- normalized sample range `[-1.0, 1.0]`

## 処理

1. frame contract を検証する。
2. 全 sample を `float` として読み、finite と normalized range を確認する。
3. `peak = max(abs(sample))` を計算する。
4. `peak < silence_threshold_linear` なら data を変更せず pass-through する。
5. それ以外は `gain = target_peak_linear / peak` を全 sample に適用する。
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

出力範囲を満たせない frame は publish せず drop する。
