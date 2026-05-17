# fa_asr

`fa_asr` は `audio/frame` を `TurnContext` 単位でバッファし、VAD終了または無音タイムアウトでローカルASR実行ファイルにWAVを渡します。OpenAI APIやWebSocketには依存しません。

## 入出力

- Sub: `audio/frame` (`fa_interfaces/msg/AudioFrame`)
- Sub: `voice/vad_state` (`fa_interfaces/msg/VadState`)
- Sub: `conversation/turn_context` (`fa_interfaces/msg/TurnContext`)
- Pub: `voice/asr/result` (`fa_interfaces/msg/AsrResult`)

## バックエンド契約

`config/default.yaml` の `backend.command` と `backend.model_path` は必須です。`backend.args` には `{audio}` と `{model}` を含めてください。

例: whisper.cpp の `whisper-cli` を使う場合

```yaml
backend.command: "whisper-cli"
backend.model_path: "/models/ggml-large-v3.bin"
backend.language: "ja"
backend.args: ["-m", "{model}", "-l", "{language}", "-f", "{audio}", "-nt"]
```
