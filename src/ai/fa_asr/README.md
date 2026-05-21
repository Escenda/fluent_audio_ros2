# fa_asr

`fa_asr` は `audio/frame` を `TurnContext` 単位でバッファし、VAD終了または無音タイムアウトで ASR backend に raw float32le payload を渡します。OpenAI SDK/APIやWebSocket client には直接依存しません。

## 入出力

- Sub: `audio/frame` (`fa_interfaces/msg/AudioFrame`)
- Sub: `conversation/turn_context` (`fa_interfaces/msg/TurnContext`)
- Sub: `control.speech_control.topic=voice/vad_state` (`fa_interfaces/msg/VadState`)
- Pub: `voice/asr/result` (`fa_interfaces/msg/AsrResult`)
- Pub: `voice/asr/state` (`fa_interfaces/msg/AsrState`)
- Pub: `voice/asr/event` (`fa_interfaces/msg/AsrEvent`)
- Srv: `transcribe_audio` (`fa_interfaces/srv/TranscribeAudio`)

`expected_source_id` / `expected_stream_id` は必須です。受信した `AudioFrame.source_id` と control message の `source_id` は `expected_source_id`、`AudioFrame.stream_id` と control message の `stream_id` は `expected_stream_id` と一致する必要があります。別 source / stream の audio frame や control close は ASR buffer に混ぜず reject します。現行既定の control は `control.speech_control.*` で、`VadState` の `is_speech` / `start` / `end` を ASR window 開閉に使います。

## QoS

QoS は edge ごとに明示します。depth は正の整数、reliable は bool として扱い、node 内で topic 名から推測しません。

```yaml
audio.qos.depth: 20
audio.qos.reliable: false
control.speech_control.qos.depth: 50
control.speech_control.qos.reliable: false
turn_context.qos.depth: 10
turn_context.qos.reliable: true
result.qos.depth: 10
result.qos.reliable: true
observability.qos.depth: 50
observability.qos.reliable: true
```

## バックエンド契約

`backend.name` は必須です。対応する backend は `local_command`, `whisper.cpp`, `parakeet_worker`, `nemo_rnnt_streaming`, `openai_realtime`, `openai_transcriptions` です。
non-streaming command 系 backend では `backend.result_format` も backend 設定時に必須です。`nemo_rnnt_streaming` は JSONL streaming protocol を使い、`backend.result_format` / `backend.args` / `backend.health_args` は使いません。default config は `backend.name` / command backend 用 `backend.result_format` を空にし、backend selection と output contract を暗黙選択しません。利用環境ごとの launch/config で `backend.name` と backend ごとの必須パラメータを明示してください。

- `local_command` / `whisper.cpp`: `backend.command` と `backend.model_path` が必須です。
- `parakeet_worker`: `backend.command`、`backend.model`、`backend.result_format`、`backend.args`、`backend.health_args` が必須です。Python version / venv / SDK が異なる処理は外部 worker / process / container 側へ置きます。streaming session、encoder cache、NeMo import、NIM/Riva endpoint、NGC artifact handling は持ちません。
- `nemo_rnnt_streaming`: `backend.command` と local `.nemo` file の `backend.model_path` が必須です。JSONL streaming worker と cache-aware NeMo RNNT model を使い、`backend.result_format` / `backend.args` / `backend.health_args` は使いません。
- `openai_realtime` / `openai_transcriptions`: `backend.command`、`backend.model`、対応する `backend.openai_*.api_key_env` が必須です。OpenAI SDK/API client は外部 worker / process / container 側へ置きます。
- `backend.args` は non-streaming command backend の worker/CLI contract です。該当 backend では `{audio}`、`{model}`、`{sample_rate}` を含む配列を明示してください。
- `backend.health_args` は non-streaming command backend の startup health check contract です。`parakeet_worker` / `openai_realtime` / `openai_transcriptions` では必須です。`local_command` / `whisper.cpp` では package 単体の backend contract としては任意ですが、package-owned SO101 profile template では startup health check を明示するために設定します。
- `backend.result_format` は non-streaming command backend の出力 contract で、`plain_text` または `segments_json_v1` です。`plain_text` は transcript text を selected ASR request samples 全体の 1 segment にします。`segments_json_v1` は `result_format` と `segments` だけを持つ strict JSON で、segment offset は selected ASR request samples からの相対 sample index です。invalid JSON/schema/range、overlap、空 text は fail closed であり、`fa_asr` は推測補正、型 coercion、resample、downmix、PCM/int16 変換を行いません。

例: whisper.cpp を raw float32le worker 経由で使う場合

```yaml
backend.name: "whisper.cpp"
backend.command: "/opt/fluent_audio/bin/whisper_cpp_worker"
backend.model_path: "/models/ggml-large-v3.bin"
backend.language: "ja"
backend.result_format: "plain_text"
backend.args: ["--model", "{model}", "--language", "{language}", "--audio-f32", "{audio}", "--sample-rate", "{sample_rate}"]
backend.health_args: ["health", "--model", "{model}", "--language", "{language}"]
```

OpenAI 系 backend は OpenAI 直結実装ではなく、外部 worker command を呼ぶ backend slot です。API key の値、network、SDK は worker 側の責務です。`fa_asr` は `api_key_env` の指定と、その環境変数が空でないことだけを検証し、未設定なら起動失敗します。
