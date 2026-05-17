# fa_trim

`fa_trim` は FluentAudio の `processing/temporal` に属する ROS2 package で、`AudioFrame`
ごとに先頭と末尾の sample frame を指定数だけ削除する。

## Topics

| direction | type | name |
| --- | --- | --- |
| subscribe | `fa_interfaces/msg/AudioFrame` | `input_topic` parameter |
| publish | `fa_interfaces/msg/AudioFrame` | `output_topic` parameter |
| publish | `diagnostic_msgs/msg/DiagnosticArray` | `diagnostics` |

## Contract

- 入力は `FLOAT32LE`、32bit、interleaved PCM に限定する。
- `stream_id` は `output_topic` に更新する。
- `source_id`、`header`、`encoding`、`sample_rate`、`channels`、`bit_depth`、`layout` は入力を継承する。
- payload が変わるため、`epoch` は `input.epoch + 1` にする。wrap する入力は drop する。
- trim 後に sample frame が残らない場合は publish せず diagnostics に記録する。

## Launch

```bash
ros2 launch fa_trim fa_trim.launch.py
```

設定例は `config/default.yaml`、詳細仕様は `docs/仕様書.md` を参照する。
