# fa_reverb

`fa_reverb` は FluentAudio の `processing/temporal` に属する ROS2 package です。
`FLOAT32LE` / `interleaved` の `fa_interfaces/msg/AudioFrame` を購読し、内部 multi-tap feedback delay network による残響を適用して publish します。sample loop と delay state は ROS 非依存 backend が持ちます。

## Topics

| 種別 | 既定 config 例 | Message |
| --- | --- | --- |
| input | `fa_reverb/input` | `fa_interfaces/msg/AudioFrame` |
| output | `fa_reverb/output` | `fa_interfaces/msg/AudioFrame` |
| diagnostics | `diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` |

## Parameters

実行時 parameter はすべて明示指定が必要です。コード上に有効な runtime default は持ちません。

- `input_topic`
- `output_topic`
- `input_stream_id`
- `output.stream_id`
- `reverb.room_size`
- `reverb.damping`
- `reverb.wet_gain`
- `reverb.dry_gain`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.bit_depth`
- `expected.layout`
- `qos.depth`
- `qos.reliable`
- `diagnostics.publish_period_ms`

## Launch

```bash
ros2 launch fa_reverb fa_reverb.launch.py
```

詳細な外部契約は [docs/仕様書.md](docs/仕様書.md)、処理内容は [docs/アルゴリズム詳細説明書.md](docs/アルゴリズム詳細説明書.md)、検証方針は [docs/テスト設計.md](docs/テスト設計.md) を参照してください。
