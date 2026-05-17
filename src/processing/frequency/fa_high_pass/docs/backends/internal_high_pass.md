# internal_high_pass backend

`fa_high_pass` の初期 backend は node 内 C++ 実装である。

## Scope

- `AudioFrame.data` を FLOAT32LE interleaved sample として処理する。
- channel ごとに一次 high-pass filter state を保持する。
- `filter.cutoff_hz` から係数を計算する。

## Non-Scope

- device API
- resample
- format conversion
- normalize / gain / limit
- external inference runtime

外部 DSP backend を追加する場合も、ROS2 node の契約は `docs/仕様書.md` を正とし、暗黙の format 変換や cutoff 補正は backend 側へ入れない。
