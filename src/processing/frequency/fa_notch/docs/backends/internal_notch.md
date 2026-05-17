# internal_notch backend

`fa_notch` の初期 backend は node 内 C++ 実装である。

## Scope

- `AudioFrame.data` を FLOAT32LE interleaved sample として処理する。
- channel ごとに二次 biquad filter state を保持する。
- `filter.center_hz` と `filter.q` から notch 係数を計算する。
- 係数は処理前に `a0` で正規化する。

## Non-Scope

- device API
- resample
- format conversion
- normalize / gain / limit
- external inference runtime

外部 DSP backend を追加する場合も、ROS2 node の契約は `docs/仕様書.md` を正とし、暗黙の format 変換や center / Q 補正は backend 側へ入れない。
