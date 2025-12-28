# FV TTS

`fv_tts`はpyopenjtalk(Open JTalk)を利用したシンプルなTTSノードです。`fv_audio/msg/AudioFrame`をPublishしつつ`fv_tts/srv/Speak`サービスで合成を提供します。

## 依存
- `pyopenjtalk`
- `python3-numpy`

## 起動
```bash
ros2 run fv_tts fv_tts_node --ros-args -p default_voice:=mei_normal
```
または
```bash
ros2 launch fv_tts fv_tts.launch.py
```

## サービス呼び出し例
```bash
ros2 service call /fv_tts/speak fv_tts/srv/Speak "{text: 'こんにちは', voice_id: '', play: true, volume_db: 0.0, cache_key: ''}"
```
レスポンスの`frame`をそのまま録音・出力ノードに渡せます。`play: true`の場合は`audio/output/frame`へPublishするため、別途`fv_audio_output`が購読していればスピーカー再生されます。

## キャッシュ
`cache_dir`（デフォルト: `~/.fluent_voice_cache`）にPCMとメタ情報を保存します。`cache_key`を指定すると同じキーでキャッシュを共有できます。
