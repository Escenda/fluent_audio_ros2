# internal_rms_agc backend

## 1. Backend

`fa_agc` の backend は node 内部の frame RMS AGC 実装である。外部 DSP library、device API、device gain API、`fa_in`、resampler、limiter、compressor、normalize backend は使わない。

## 2. 入力

- `fa_interfaces/msg/AudioFrame`
- `FLOAT32LE`
- `interleaved`
- finite normalized sample

## 3. 処理

frame RMS を計算し、`target_rms / frame_rms` を `min_gain` から `max_gain` に制限した target gain に変換する。gain を下げる場合は attack、gain を上げる場合は release の time constant から smoothing coefficient を計算する。

## 4. 出力

`AudioFrame.data` を AGC 後 sample bytes に更新する。出力 sample が正規化範囲を超える場合は frame を publish せず、candidate gain も確定しない。
