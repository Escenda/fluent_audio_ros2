# fa_packet_loss_concealment

`fa_packet_loss_concealment` は、FluentAudio の `streaming` に属する明示的な packet loss concealment node です。

入力 `AudioFrame` の `epoch` が欠落した場合だけ、直前の有効 frame を `plc.attenuation_per_gap` で減衰して最大 `plc.max_gap_frames` 枚まで合成します。これは隠れた fallback ではなく、system config でこの node を pipeline に挟んだ場合だけ有効になる PLC 処理です。

## Topics

| 種別 | topic | message |
| --- | --- | --- |
| subscribe | `input_topic` | `fa_interfaces/msg/AudioFrame` |
| publish | `output_topic` | `fa_interfaces/msg/AudioFrame` |
| publish | `diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` |

## 起動例

```bash
ros2 launch fa_packet_loss_concealment fa_packet_loss_concealment.launch.py \
  node_name:=fa_packet_loss_concealment_node \
  config_file:=/path/to/fa_packet_loss_concealment.yaml
```

## 主要 config

- `input_topic`
- `output_topic`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.bit_depth`
- `expected.layout`
- `plc.max_gap_frames`
- `plc.attenuation_per_gap`
- `qos.depth`
- `qos.reliable`
- `diagnostics.publish_period_ms`

詳細は `docs/仕様書.md`、`docs/アルゴリズム詳細説明書.md`、`docs/テスト設計.md`、`docs/backends/repeat_attenuation_plc.md` を参照してください。
