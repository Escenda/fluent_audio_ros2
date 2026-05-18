# FA Stream

`fa_stream`は`audio/frame`（`fa_interfaces/msg/AudioFrame`）を購読し、`ffmpeg`へパイプしてIcecast/Shoutcast系エンドポイントへ送る network stream sink utility です（クラウド依存はなく、任意のHTTP PUT先へ送出できます）。

この package は network sink であり、`src/streaming` の jitter buffer / clock drift / PLC などのリアルタイム伝送安定化 node ではありません。

## 起動
```bash
ros2 launch fa_stream fa_stream.launch.py \
  node_name:=fa_stream \
  config_file:=/path/to/fa_stream.yaml
```

## 主なパラメータ
- `input_topic`（default: `audio/frame`）
- `ffmpeg_path`（default: `ffmpeg`）
- `output_url`（必須。空なら起動失敗）
- `audio_codec` / `bitrate` / `container_format` / `content_type`

注: 現状は16bit PCMのみ対応です（`AudioFrame.bit_depth == 16`）。
