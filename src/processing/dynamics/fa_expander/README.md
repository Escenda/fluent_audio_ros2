# fa_expander

`fa_expander` は FluentAudio の `fa_interfaces/msg/AudioFrame` を入力し、FLOAT32LE interleaved の正規化済みサンプルに静的な下向きエキスパンダを適用する ROS2 パッケージです。

## 入出力

- executable: `fa_expander_node`
- input topic: `fa_expander/input`
- output topic: `fa_expander/output`
- input stream: `audio/noise_gated/mic`
- output stream: `audio/expanded/mic`
- encoding: `FLOAT32LE`
- bit depth: `32`
- layout: `interleaved`

入力 `AudioFrame` は `source_id` と `stream_id` が空でなく、`stream_id` が `input_stream_id` と一致する必要があります。出力は入力フレームの header、source_id、sample_rate、channels、bit_depth、encoding、layout、epoch を保持し、`stream_id` は `output.stream_id` に更新します。

## パラメータ

- `expander.threshold_linear`: しきい値。有限、`0.0 < value < 1.0`。
- `expander.ratio`: 下向き展開比。有限、`value > 1.0`。

不正な設定では起動時に fail closed します。実行時に契約違反または非正規化サンプルを含むフレームを受け取った場合、そのフレームを drop し warning を出します。

```bash
ros2 launch fa_expander fa_expander.launch.py node_name:=fa_expander config_file:=/path/to/fa_expander.yaml
```
