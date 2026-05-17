# internal_static_curve backend

## 1. Backend

`fa_compressor` の backend は node 内部の static curve 実装である。外部 DSP library、device API、resampler、limiter、gate、normalize backend は使わない。

## 2. 入力

- `fa_interfaces/msg/AudioFrame`
- `FLOAT32LE`
- `interleaved`
- finite normalized sample

## 3. 処理

sample ごとに `threshold + (abs(sample) - threshold) / ratio` の knee なし静的圧縮を行い、符号を戻して makeup gain を掛ける。

## 4. 出力

`AudioFrame.data` を圧縮後 sample bytes に更新する。出力 sample が正規化範囲を超える場合は frame を publish しない。
