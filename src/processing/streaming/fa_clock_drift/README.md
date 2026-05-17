# fa_clock_drift

`fa_clock_drift` は `fa_interfaces/msg/AudioFrame` の `header.stamp` を、sample-clock drift の推定値に基づいて bounded correction する C++ ROS2 processing node です。

この package は device I/O、sample rate conversion、drop/add samples、padding、normalization、payload の編集を行いません。入力 `AudioFrame.data` は byte 列として保持し、出力へ同一内容を渡します。

## Topics

| Direction | Topic | Type | Description |
| --- | --- | --- | --- |
| Subscribe | `input_topic` | `fa_interfaces/msg/AudioFrame` | drift 補正前の音声 frame |
| Publish | `output_topic` | `fa_interfaces/msg/AudioFrame` | timestamp 補正後の音声 frame |
| Publish | `diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | counters、設定値、drift 推定値 |

## Parameters

| Parameter | Required | Description |
| --- | --- | --- |
| `input_topic` | yes | 入力 `AudioFrame` topic。入力 frame の `stream_id` と一致すること |
| `output_topic` | yes | 出力 `AudioFrame` topic。出力 frame の `stream_id` に設定する値 |
| `expected.sample_rate` | yes | 期待する sample rate |
| `expected.channels` | yes | 期待する channel 数 |
| `expected.encoding` | yes | 期待する encoding |
| `expected.bit_depth` | yes | 期待する bit depth |
| `expected.layout` | yes | 期待する layout |
| `drift.ema_alpha` | yes | observed drift に対する EMA 係数。`0.0 < alpha <= 1.0` |
| `drift.max_correction_ms_per_frame` | yes | 1 frame あたりの補正上限 milliseconds |
| `drift.reset_threshold_ms` | yes | observed drift がこの絶対値を超えたら timeline reset |
| `qos.depth` | yes | subscription / publisher QoS depth |
| `qos.reliable` | yes | true なら reliable、false なら best effort |
| `diagnostics.publish_period_ms` | yes | diagnostics publish period |

## Fail-Closed Behavior

起動時に必須 parameter が欠けている、または drift parameter が有限値・範囲条件を満たさない場合は node を起動しません。

runtime では `source_id` / `stream_id`、format fields、non-empty data、frame byte alignment を検証します。契約違反 frame は drift timeline を変更する前に drop します。補正後 timestamp が負、非有限、または `builtin_interfaces/Time` の範囲外になる場合は frame を publish せず、timeline state を reset します。
