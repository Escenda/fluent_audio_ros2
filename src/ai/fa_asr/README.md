# fa_asr

`fa_asr` は `audio/frame` を `TurnContext` 単位でバッファし、VAD終了または無音タイムアウトで ASR backend に raw float32le payload を渡します。OpenAI SDK/APIやWebSocket client には直接依存しません。

## 入出力

- Sub: `audio/frame` (`fa_interfaces/msg/AudioFrame`)
- Sub: `voice/vad_state` (`fa_interfaces/msg/VadState`)
- Sub: `conversation/turn_context` (`fa_interfaces/msg/TurnContext`)
- Pub: `voice/asr/result` (`fa_interfaces/msg/AsrResult`)

`expected_source_id` / `expected_stream_id` は必須です。受信した `AudioFrame.source_id` と `VadState.source_id` は `expected_source_id`、`AudioFrame.stream_id` と `VadState.stream_id` は `expected_stream_id` と一致する必要があります。別 source / stream の audio frame や VAD end は ASR buffer に混ぜず reject します。

## バックエンド契約

`backend.name` は必須です。対応する backend は `local_command`, `whisper.cpp`, `parakeet_worker`, `openai_realtime`, `openai_transcriptions` です。
default config は backend を暗黙選択しません。利用環境ごとの launch/config で `backend.name` と必須パラメータを明示してください。

- `local_command` / `whisper.cpp`: `backend.command` と `backend.model_path` が必須です。
- `parakeet_worker`: `backend.command` と `backend.model` が必須です。Python version / venv / SDK が異なる処理は外部 worker / process / container 側へ置きます。
- `openai_realtime` / `openai_transcriptions`: `backend.command`、`backend.model`、対応する `backend.openai_*.api_key_env` が必須です。OpenAI SDK/API client は外部 worker / process / container 側へ置きます。
- `backend.args` は default config では空です。backend ごとの worker/CLI contract として `{audio}`、`{model}`、`{sample_rate}` を含む配列を明示してください。

例: whisper.cpp を raw float32le worker 経由で使う場合

```yaml
backend.name: "whisper.cpp"
backend.command: "/opt/fluent_audio/bin/whisper_cpp_worker"
backend.model_path: "/models/ggml-large-v3.bin"
backend.language: "ja"
backend.args: ["--model", "{model}", "--language", "{language}", "--audio-f32", "{audio}", "--sample-rate", "{sample_rate}"]
```

OpenAI 系 backend は OpenAI 直結実装ではなく、外部 worker command を呼ぶ backend slot です。API key の値、network、SDK は worker 側の責務です。`fa_asr` は `api_key_env` の指定と、その環境変数が空でないことだけを検証し、未設定なら起動失敗します。
