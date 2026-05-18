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
- 入力 `stream_id` は `input_stream_id` と一致必須。
- 出力 `stream_id` は `output.stream_id` に更新する。
- `source_id`、`header`、`encoding`、`sample_rate`、`channels`、`bit_depth`、`layout` は入力を継承する。
- payload が変わるため、`epoch` は `input.epoch + 1` にする。wrap する入力は drop する。
- trim 後に sample frame が残らない場合は publish せず diagnostics に記録する。
- trim 本体と sample 検証は ROS2 非依存の `internal_frame_trim` backend が担当する。

## Launch

```bash
ros2 launch fa_trim fa_trim.launch.py config_file:=/path/to/fa_trim.yaml
```

`config_file` は必須。設定例は `config/default.yaml`、詳細仕様は `docs/仕様書.md` を参照する。
