# fa_time_alignment

`fa_time_alignment` は `fa_interfaces/msg/AudioFrame` の `header.stamp` を明示的な時間グリッドへ揃える streaming node です。

このノードは音声データを編集しません。入力フレームを検証し、許容範囲内の時刻補正だけを `header.stamp` に反映し、出力ストリームの `stream_id` に更新して publish します。

## 入出力

- 入力: `fa_interfaces/msg/AudioFrame`
- 出力: `fa_interfaces/msg/AudioFrame`
- diagnostics: `diagnostic_msgs/msg/DiagnosticArray`

## 主なパラメータ

- `input_topic`: 必須。入力 `AudioFrame` の topic。
- `output_topic`: 必須。出力 `AudioFrame` の topic。
- `expected.*`: 入力フレームの sample rate、channels、encoding、bit depth、layout。
- `alignment.period_ms`: 時間グリッド周期。有限かつ `> 0`。
- `alignment.phase_ms`: グリッド位相。有限かつ `>= 0`、`period_ms` 未満。
- `alignment.max_adjust_ms`: 許容する最大時刻補正量。有限かつ `>= 0`。
- `qos.depth`, `qos.reliable`: 入出力 QoS。
- `diagnostics.publish_period_ms`: diagnostics publish 周期。

## fail-closed 条件

次の場合は frame を publish せず drop します。

- `source_id` または `stream_id` が空
- `stream_id` が `input_topic` と一致しない
- format fields が `expected.*` と一致しない
- `data` が空、または 1 audio frame の byte 境界に揃っていない
- nearest grid timestamp が負
- 必要な補正量の絶対値が `alignment.max_adjust_ms` を超える

## 非責務

- device I/O
- resampling
- channel conversion
- encoding conversion
- audio sample editing
- stale timestamp を現在値として扱う fallback
