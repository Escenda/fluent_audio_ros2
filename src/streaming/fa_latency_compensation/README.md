# fa_latency_compensation

`fa_latency_compensation` は `fa_interfaces/msg/AudioFrame` の `header.stamp` だけを設定値で補正する C++ ROS2 processing node です。

この package は device I/O、sample 変換、gain、filter、resample、buffering を行いません。入力 `AudioFrame.data` は byte 列として保持し、出力へ同一内容を渡します。

## Topics

| Direction | Topic | Type | Description |
| --- | --- | --- | --- |
| Subscribe | `input_topic` | `fa_interfaces/msg/AudioFrame` | 補正前の音声 frame |
| Publish | `output_topic` | `fa_interfaces/msg/AudioFrame` | timestamp 補正後の音声 frame |
| Publish | `diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | counters と設定値 |

## Parameters

| Parameter | Required | Description |
| --- | --- | --- |
| `input_topic` | yes | 入力 `AudioFrame` topic。入力 frame の `stream_id` と一致すること |
| `output_topic` | yes | 出力 `AudioFrame` topic。出力 frame の `stream_id` に設定する値 |
| `compensation.offset_ms` | yes | `header.stamp` に加算する signed double milliseconds |
| `expected.sample_rate` | yes | 期待する sample rate |
| `expected.channels` | yes | 期待する channel 数 |
| `expected.encoding` | yes | 期待する encoding |
| `expected.bit_depth` | yes | 期待する bit depth |
| `expected.layout` | yes | 期待する layout |
| `qos.depth` | yes | subscription / publisher QoS depth |
| `qos.reliable` | yes | true なら reliable、false なら best effort |
| `diagnostics.publish_period_ms` | yes | diagnostics publish period |

## Fail-Closed Behavior

起動時に必須 parameter が欠けている、または `compensation.offset_ms` が有限値でない場合は node を起動しません。

runtime では `source_id` / `stream_id`、format fields、non-empty data、frame byte alignment を検証します。契約違反 frame と、補正後 timestamp が負になる frame は publish せず drop します。
