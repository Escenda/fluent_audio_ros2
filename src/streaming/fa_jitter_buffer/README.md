# fa_jitter_buffer

`fa_jitter_buffer` は `fa_interfaces::msg::AudioFrame` を epoch 順に並べ、指定した frame 深さを超えた時点で最古の frame から publish する streaming processing node である。

この package は network jitter の吸収だけを担当する。音声 payload の編集、resample、padding、欠損 frame の生成、device I/O は行わない。

## Node

| item | value |
| --- | --- |
| package | `fa_jitter_buffer` |
| executable | `fa_jitter_buffer_node` |
| default node name | `fa_jitter_buffer_node` |
| class | `fa_jitter_buffer::FaJitterBufferNode` |

## Parameters

すべて起動時必須であり、コード内の有効 default は持たない。

| parameter | meaning |
| --- | --- |
| `input_topic` | subscribe する `AudioFrame` topic |
| `output_topic` | publish する `AudioFrame` topic |
| `expected.sample_rate` | 受け入れる sample rate |
| `expected.channels` | 受け入れる channel 数 |
| `expected.encoding` | 受け入れる encoding |
| `expected.bit_depth` | 受け入れる bit depth |
| `expected.layout` | 受け入れる layout |
| `jitter.target_depth_frames` | 出力前に残す buffer 深さ |
| `jitter.max_depth_frames` | 設定上許容する最大 buffer 深さ。`target_depth_frames` より大きい必要がある |
| `jitter.reset_on_epoch_regression` | publish 済み epoch より古い epoch を stream reset として扱うか |
| `qos.depth` | AudioFrame topic QoS depth |
| `qos.reliable` | reliable QoS を使うか |
| `diagnostics.publish_period_ms` | diagnostics publish 周期 |

## Runtime Contract

- `source_id` / `stream_id` が空の frame は drop する。
- `stream_id` が `input_topic` と一致しない frame は drop する。
- format が `expected.*` と完全一致しない frame は drop する。
- data が空、または sample frame byte 境界に揃わない frame は drop する。
- `FLOAT32LE` + `interleaved` の場合は全 sample が finite かつ `[-1.0, 1.0]` に収まることを検証する。
- 受理した frame は epoch を key にした buffer へ入れる。
- buffer size が `jitter.target_depth_frames` を超える間、最古 epoch の frame を publish する。
- duplicate epoch は drop する。
- publish 済み epoch より古い epoch は late drop する。ただし `reset_on_epoch_regression=true` の場合は buffer を reset して新しい epoch sequence として受け入れる。
- `source_id` または format contract が変わった場合は buffer を reset し、別 stream として扱う。

## Diagnostics

`diagnostics` topic へ次の値を publish する。

- config: input/output topic、expected format、target/max depth、reset policy
- state: `buffered_frames`
- counters: `frames_in`, `frames_out`, `frames_dropped`, `duplicate_drops`, `late_drops`, `resets`
