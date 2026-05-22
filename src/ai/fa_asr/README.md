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

`backend.name` は必須です。標準 ASR path は `parakeet_multilingual_buffered` です。

`parakeet_multilingual_buffered` は、multilingual Parakeet 1.1B の local `.nemo` model を `fa_asr` backend/session 内部の runner として保持します。外部 process、stdin/stdout JSONL worker、NIM、Riva、gRPC、Whisper、OpenAI へ fallback しません。

この backend の入力 contract は固定です。

- `backend.model_path`: local multilingual Parakeet 1.1B `.nemo`
- `backend.language`: 空文字
- `backend.language_policy`: `auto_detect`
- `backend.sample_rate_hz`: `16000`
- `backend.channels`: `1`
- `backend.chunk_size_samples` または `backend.chunk_ms`: partial re-decode cadence
- `backend.emit_partial`: partial hypothesis を publish するか
- `backend.max_buffer_sec`: rolling buffer retention
- `backend.speech_energy_threshold`: speech audio に対する empty final rejection threshold

`fa_asr` は hidden resample、downmix、PCM/int16 変換を行いません。ASR backend へ渡される audio は upstream pipeline で `FLOAT32LE` / 16 kHz / mono にそろえられている必要があります。未対応 encoding、sample rate、channel は backend decode 前に fail closed します。

Parakeet 1.1B multilingual `.nemo` は full-context model として扱います。cache-aware streaming model ではないため、FluentAudio では rolling buffer を chunk 境界で再 decode し、partial を未確定 hypothesis として返します。VAD / TurnDetector / timeout などで stream が閉じたとき、`finish()` が final decode を行い、その結果だけを committed transcript とします。

speech energy が十分ある audio に対して final transcript が空なら成功扱いしません。英語専用 model、Whisper、旧 JSONL worker、外部 serving stack へ自動 fallback もしません。
