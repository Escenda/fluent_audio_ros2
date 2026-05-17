# fa_asr

`fa_asr` は `audio/frame` を `TurnContext` 単位でバッファし、VAD終了または無音タイムアウトで ASR backend にWAVを渡します。OpenAI SDK/APIやWebSocket client には直接依存しません。

## 入出力

- Sub: `audio/frame` (`fa_interfaces/msg/AudioFrame`)
- Sub: `voice/vad_state` (`fa_interfaces/msg/VadState`)
- Sub: `conversation/turn_context` (`fa_interfaces/msg/TurnContext`)
- Pub: `voice/asr/result` (`fa_interfaces/msg/AsrResult`)

## バックエンド契約

`backend.name` は必須です。対応する backend は `local_command`, `whisper_cpp`, `parakeet_worker`, `openai_realtime` です。

- `local_command` / `whisper_cpp`: `backend.command` と `backend.model_path` が必須です。
- `parakeet_worker` / `openai_realtime`: `backend.command` と `backend.model` が必須です。Python version / venv / SDK が異なる処理は外部 worker / process / container 側へ置きます。
- `backend.args` には `{audio}` と `{model}` を含めてください。

例: whisper.cpp の `whisper-cli` を使う場合

```yaml
backend.command: "whisper-cli"
backend.model_path: "/models/ggml-large-v3.bin"
backend.language: "ja"
backend.args: ["-m", "{model}", "-l", "{language}", "-f", "{audio}", "-nt"]
```

`openai_realtime` は OpenAI 直結実装ではなく、外部 worker command を呼ぶ backend slot です。API key、network、SDK は worker 側の責務で、`fa_asr` は未設定なら起動失敗します。
