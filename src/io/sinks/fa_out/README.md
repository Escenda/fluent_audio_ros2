# FA Out

`fa_out`は`fa_interfaces/msg/AudioFrame`を明示 backend で選んだ sink に出力するROS2ノードです。`input_topic`で指定したtopicを購読し、`input_stream_id`に一致する frame だけを speaker device、raw PCM file、または raw PCM UDP endpoint へ書き込みます。

## 依存
- ALSA (`libasound2-dev`)
- `fa_interfaces`メッセージ

## 起動
```bash
ros2 launch fa_out fa_out.launch.py node_name:=fa_out config_file:=/path/to/fa_out.yaml
```

`config/default.yaml` は `alsa_playback` backend 用で、site 固有の sink id を空にしています。
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
- `backend.name`: `alsa_playback` / `pcm_file_writer` / `network_pcm_sender`
- `audio.device_id`: ALSA raw hardware device id（`alsa_playback`、例: `hw:1,0`）
- `file.path`: raw PCM output path（`pcm_file_writer`）
- `overwrite.enabled`: 既存 file を上書きするか（`pcm_file_writer`）
- `endpoint.uri`: raw PCM UDP endpoint（`network_pcm_sender`、例: `udp://127.0.0.1:9000`）
- `transport.identity`: network transport identity（`network_pcm_sender`）
- `network.max_packet_bytes`: 1 UDP packet の最大 byte 数（`network_pcm_sender`）
- `audio.sample_rate`, `audio.channels`, `audio.bit_depth`: フレームと一致している必要があります。
- `queue.max_frames`: バッファに保持するフレーム数。溢れた場合は frame drop で継続せず fail closed します。

`fa_tts`と組み合わせる場合、`fa_tts` の出力を `fa_mix` などの routing node で `input_topic` に流し、routing 後の `AudioFrame.stream_id` を `input_stream_id` に揃えてから再生します。

`pcm_file_writer` は raw bytes を書くだけで、WAV/MP3/AAC/FLAC container 生成、encode、resample、gain は行いません。`network_pcm_sender` は accepted `AudioFrame` 1 件を 1 UDP packet として送るだけで、jitter buffer、PLC、clock drift correction は行いません。必要な処理は `fa_encode`、`fa_sample_format`、`fa_gain`、`src/streaming` の各 node などを前段に置きます。
