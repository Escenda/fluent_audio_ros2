# FA Stream

`fa_stream`は`audio/frame`（`fa_interfaces/msg/AudioFrame`）を購読し、`ffmpeg`へパイプしてIcecast/Shoutcast系エンドポイントへ送るサンプルです（クラウド依存はなく、任意のHTTP PUT先へ送出できます）。

## 起動
```bash
ros2 launch fa_stream fa_stream.launch.py
```

## 主なパラメータ
- `input_topic`（default: `audio/frame`）
- `ffmpeg_path`（default: `ffmpeg`）
- `output_url`（default: `http://source:hackme@localhost:8000/live`）
- `audio_codec` / `bitrate` / `container_format` / `content_type`

注: 現状は16bit PCMのみ対応です（`AudioFrame.bit_depth == 16`）。

