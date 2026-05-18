# internal_notch backend

`fa_notch` の初期 backend は ROS2 非依存の C++ 実装 `fa_notch::backends::InternalNotchBackend` である。

## Scope

- FLOAT32LE interleaved sample bytes を処理する。
- channel ごとに二次 biquad filter state を保持する。
- `filter.center_hz` と `filter.q` から notch 係数を計算する。
- 係数は処理前に `a0` で正規化する。
- backend は ROS2 topic、ROS message、`rclcpp` を参照しない。

## Failure Contract

- 空入力は `kEmptyInput` として拒否する。
- `channels * sizeof(float)` に揃わない byte 長は `kMisalignedInput` として拒否する。
- non-finite input sample は `kNonFiniteInput` として拒否する。
- FLOAT32LE として表現できない output sample は `kNonFiniteOutput` として拒否する。
- 拒否時は channel filter state を更新しない。

## Non-Scope

- device API
- resample
- format conversion
- normalize / gain / limit
- external inference runtime

外部 DSP backend を追加する場合も、ROS2 node の契約は `docs/仕様書.md` を正とし、暗黙の format 変換や center / Q 補正は backend 側へ入れない。
