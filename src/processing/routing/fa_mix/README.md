# fa_mix

複数入力のミキシング（MVP）ノードです。現状は **PCM16LE の加算ミックス + ゲイン**のみを実装しています。
PCM16LE decode、gain application、mix、overflow detection、PCM16LE encode は ROS-free backend `internal_pcm16_mixer` が担当します。node は ROS topic、latest frame cache、timestamp stale 判定、metadata wrapping、diagnostics を担当します。
configured input が欠ける、stale、decode 不能、または sample count mismatch の場合は partial mix を publish せず frame 全体を drop します。

## Subscribe / Publish
- Sub: `input_topics`（`fa_interfaces/msg/AudioFrame`）
- Pub: `output_topic`（`fa_interfaces/msg/AudioFrame`）

`input_topics` / `output_topic` は ROS topic 名、`input_stream_ids` / `output.stream_id` は `AudioFrame.stream_id` です。両者は一致させません。

## Run
```bash
ros2 launch fa_mix fa_mix.launch.py node_name:=fa_mix config_file:=/path/to/fa_mix.yaml
```

`node_name` と `config_file` はどちらも明示必須です。設定例は `config/default.yaml` を参照してください。
