# FA Out

`fa_out`は`fa_interfaces/msg/AudioFrame`をALSA raw hardware device に出力するROS2ノードです。`input_topic`で指定したtopicを購読し、`input_stream_id`に一致するPCM16LE frameだけをスピーカーへ再生します。

## 依存
- ALSA (`libasound2-dev`)
- `fa_interfaces`メッセージ

## 起動
```bash
ros2 launch fa_out fa_out.launch.py node_name:=fa_out config_file:=/path/to/fa_out.yaml
```

`config/default.yaml` は site 固有の sink id を空にしています。
`audio.device_id` を明示しない起動は fail closed します。
`node_name` と `config_file` は launch 時に明示する必要があります。
受信した `AudioFrame.stream_id` は `input_stream_id` と一致する必要があります。topic subscription だけを根拠に、別 stream の frame を speaker sink へ流しません。
再生完了は `playback_done_topic` に `fa_interfaces/msg/PlaybackDone` として publish します。
出力制御は `playback_control_service` の `fa_interfaces/srv/PlaybackControl` で `stop` / `pause` / `resume` を受けます。

主なパラメータ:
- `input_topic`: playback 対象 frame を受け取る ROS topic
- `input_stream_id`: speaker sink に流せる `AudioFrame.stream_id`
- `playback_done_topic`: playback 完了通知を publish する ROS topic
- `playback_control_service`: stop / pause / resume を受ける ROS service
- `audio.device_id`: ALSA raw hardware device id（例: `hw:1,0`）
- `audio.sample_rate`, `audio.channels`, `audio.bit_depth`: フレームと一致している必要があります。
- `queue.max_frames`: バッファに保持するフレーム数。溢れた場合は frame drop で継続せず fail closed します。

`fa_tts`と組み合わせる場合、`fa_tts` の出力を `fa_mix` などの routing node で `input_topic` に流し、routing 後の `AudioFrame.stream_id` を `input_stream_id` に揃えてから再生します。

ファイル出力やファイル再生は `fa_out` では扱いません。必要な場合は file sink/source 専用 package として切り出します。
