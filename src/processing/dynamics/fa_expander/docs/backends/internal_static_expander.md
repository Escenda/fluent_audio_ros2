# internal_static_expander

## Backend Scope

`internal_static_expander` は ROS 非依存の、フレーム内サンプル単位の静的下向きエキスパンダである。ROS2 topic、`fa_interfaces/msg/AudioFrame`、parameter、diagnostics、publisher/subscriber は知らない。

入力は `FLOAT32LE` interleaved sample bytes、出力は展開処理後の sample bytes である。`ProcessResult` は処理 status と `samples_expanded` を返す。

この backend は以下を行わない。

- デバイス入出力
- リサンプリング
- sample format 変換
- channel 変換
- compressor / limiter / gate / normalize / filter / denoise

## Failure Policy

設定不正は constructor で例外により fail closed する。空入力、frame boundary 不整合、非 finite sample、正規化範囲外 sample、出力不正は `ProcessStatus` として返す。

backend は拒否時に output vector を更新しない。warning、drop counter、publish 抑止は ROS node 側の責務である。
