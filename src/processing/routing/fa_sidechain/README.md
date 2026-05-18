# fa_sidechain

`fa_sidechain` は、sidechain 用 `AudioFrame` を解析し、下流 routing / dynamics node が読むための明示的な gain-control `AudioFrame` を publish する ROS2 C++ package です。

この package は program audio を購読せず、program audio の sample も変更しません。入力 sidechain frame の RMS を ROS-free backend で計算し、閾値以上なら `detector.active_gain_db`、閾値未満なら `detector.inactive_gain_db` を linear gain に変換して、mono FLOAT32LE の control frame として出力します。

## Topics

| direction | parameter | type |
| --- | --- | --- |
| subscribe | `sidechain_topic` | `fa_interfaces/msg/AudioFrame` |
| publish | `control_topic` | `fa_interfaces/msg/AudioFrame` |
| publish | `diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` |

`sidechain_topic` / `control_topic` は ROS topic 名であり、`sidechain_stream_id` / `control.stream_id` は `AudioFrame.stream_id` です。両者は一致させません。

## Input Contract

入力 sidechain frame は以下を満たす必要があります。満たさない frame は warning と diagnostics counter を残して drop します。

- `source_id` が空でない
- `stream_id == sidechain_stream_id`
- `sample_rate == expected.sample_rate`
- `channels == expected.channels`
- `encoding == FLOAT32LE`
- `bit_depth == 32`
- `layout == interleaved`
- `data` が空でなく、interleaved frame 境界に揃っている
- 全 sample が finite かつ `[-1.0, 1.0]`

## Output Contract

出力 control frame は入力 header と epoch を引き継ぎ、次の固定形式で publish します。

- `source_id = input.source_id`
- `stream_id = control.stream_id`
- `sample_rate = control.sample_rate`
- `channels = 1`
- `encoding = FLOAT32LE`
- `bit_depth = 32`
- `layout = interleaved`
- `data = float32 target_gain_linear` 1 sample

target gain は finite かつ `[0.0, 4.0]` の範囲だけを許可します。resample、normalize、clamp は行いません。

## Launch

```bash
ros2 launch fa_sidechain fa_sidechain.launch.py node_name:=fa_sidechain config_file:=/path/to/fa_sidechain.yaml
```

`node_name` と `config_file` はどちらも明示必須です。設定例は `config/default.yaml` を参照してください。
