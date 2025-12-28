# FA TTS

`fa_tts`はpyopenjtalk(Open JTalk)を利用したシンプルなTTSノードです。`fa_interfaces/msg/AudioFrame`をPublishしつつ`fa_interfaces/srv/Speak`サービスで合成を提供します。

## 依存
- `pyopenjtalk`
- `python3-numpy`

## 起動
```bash
ros2 run fa_tts fa_tts_node --ros-args -p default_voice:=mei_normal
```
または
```bash
ros2 launch fa_tts fa_tts.launch.py
```

## サービス呼び出し例
```bash
ros2 service call /speak fa_interfaces/srv/Speak "{text: 'こんにちは', voice_id: '', play: true, volume_db: 0.0, cache_key: ''}"
```
レスポンスの`frame`をそのまま録音・出力ノードに渡せます。`play: true`の場合は`audio/output/frame`へPublishするため、別途`fa_output`が購読していればスピーカー再生されます。

## キャッシュ
`cache_dir`（デフォルト: `~/.fluent_voice_cache`）にPCMとメタ情報を保存します。`cache_key`を指定すると同じキーでキャッシュを共有できます。
