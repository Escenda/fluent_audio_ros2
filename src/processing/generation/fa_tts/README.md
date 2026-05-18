# FA TTS

`fa_tts`はpyopenjtalk(Open JTalk)を利用したTTS生成ノードです。`fa_interfaces/msg/AudioFrame`を`audio/tts/frame`へPublishし、`fa_interfaces/srv/Speak`サービスで合成を提供します。

## 依存
- `python3-numpy`

`pyopenjtalk` は ROS package dependency ではなく、node 実行環境に明示的に provision します。

## 起動
```bash
ros2 launch fa_tts fa_tts.launch.py \
  node_name:=fa_tts \
  config_file:=/absolute/path/to/fa_tts.yaml
```

## サービス呼び出し例
```bash
ros2 service call /speak fa_interfaces/srv/Speak "{text: 'こんにちは', voice_id: '', play: false, volume_db: 0.0, cache_key: ''}"
```
`fa_tts` は `audio/output/frame` へ直接 publish しません。再生する場合は `audio/tts/frame` を `fa_mix` などの routing node に通し、`fa_out` へ接続します。`play: true` と `volume_db != 0.0` は generation node の責務外として拒否します。

## キャッシュ
`cache_dir`（デフォルト: `~/.cache/fluent_audio/tts`）にPCMとメタ情報を保存します。`cache_key`を指定すると同じキーでキャッシュを共有できます。
