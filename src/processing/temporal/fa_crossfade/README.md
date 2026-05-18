# fa_crossfade

`fa_crossfade` は2つの完全な FLOAT32LE interleaved `AudioFrame` segment を、明示的な overlap と fade curve で接続する temporal processing package です。

## Contract

- 入力: `input_a_topic` / `input_b_topic` の `fa_interfaces/msg/AudioFrame`
- 出力: `output_topic` の `fa_interfaces/msg/AudioFrame`
- 入力 frame は完全な segment として扱う
- `input_a` と `input_b` は同じ `epoch` の adjacent segment pair として扱う
- `source_id` と format metadata は両入力で一致必須
- `stream_id` はそれぞれ `input_a_stream_id` / `input_b_stream_id` と一致必須
- 出力 `stream_id` は `output.stream_id` に更新する
- backend は ROS2 topic/message/diagnostics を知らず、FLOAT32LE bytes だけを処理する

## Launch

```bash
ros2 launch fa_crossfade fa_crossfade.launch.py config_file:=/path/to/fa_crossfade.yaml
```

package launch の `config_file` は必須です。`config/default.yaml` は設定例であり、package launch から暗黙には読み込みません。

この node は segment の時間方向接続だけを扱い、device I/O、resample、gain normalize、routing mixer、missing segment 補完、boundary 推測は扱いません。
