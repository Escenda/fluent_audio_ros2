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

`backend.name` は必須です。対応する backend は `local_command`, `whisper.cpp`, `parakeet_worker`, `nemo_offline_transcribe`, `nemo_rnnt_streaming`, `openai_realtime`, `openai_transcriptions` です。
`nemo_offline_transcribe` は local `.nemo` を NeMo offline / full-context `transcribe(...)` API で呼ぶ non-streaming command / worker backend であり、working tree 上では backend / worker / tests / profile work が作成済みです。PO 検証では real worker health、raw `FLOAT32LE` fixture transcription、opt-in file-source full ROS graph smoke が通過済みです。ただし accuracy 評価、`TranscribeAudio` service integration、NIM / Riva / gRPC backend readiness、generic live microphone ASR は未検証です。
non-streaming command 系 backend では `backend.result_format` も backend 設定時に必須です。`nemo_rnnt_streaming` は JSONL streaming protocol を使い、`backend.result_format` / `backend.args` / `backend.health_args` は使いません。default config は `backend.name` / command backend 用 `backend.result_format` を空にし、backend selection と output contract を暗黙選択しません。利用環境ごとの launch/config で `backend.name` と backend ごとの必須パラメータを明示してください。

- `local_command` / `whisper.cpp`: `backend.command` と `backend.model_path` が必須です。
- `parakeet_worker`: `backend.command`、`backend.model`、`backend.result_format`、`backend.args`、`backend.health_args` が必須です。Python version / venv / SDK が異なる処理は外部 worker / process / container 側へ置きます。streaming session、encoder cache、NeMo import、NIM/Riva endpoint、NGC artifact handling は持ちません。
- `nemo_rnnt_streaming`: `backend.command`、local `.nemo` file の `backend.model_path`、`backend.language`、`backend.timeout_sec`、`backend.sample_rate_hz`、`backend.channels=1`、`backend.chunk_size_samples`、`backend.chunk_ms=0`、`backend.emit_partial`、`backend.max_partial_interval_ms` が必須です。JSONL streaming worker と cache-aware NeMo RNNT model を使い、`backend.result_format` / `backend.args` / `backend.health_args` は使いません。現行 Parakeet NGC artifact は finite attention context 不成立により、この local streaming backend としては未成立です。
- `nemo_offline_transcribe`: `backend.command`、local `.nemo` file の `backend.model_path`、`backend.language`、`backend.sample_rate_hz`、`backend.channels`、`backend.result_format` が必須です。NIM / Riva / gRPC は使わず、`fa_asr` node 本体は hidden resample / downmix を行いません。worker は validated raw `float32le` を NeMo model API へ渡すための WAV bridge を backend 境界内で明示的に持ちます。
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

## NeMo local `.nemo` の現 runtime evidence

`fluent-audio-runtime` container で Torch `2.11.0+cu130`、CUDA available の状態を確認し、local Parakeet 1.1B multilingual `.nemo` を `ASRModel.restore_from(...)` で `EncDecRNNTBPEModel` として restore できることを確認しています。
対象 file は `models/nemo_rnnt_streaming/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal_vtrainable_v1.0/Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo`、size `4011233560`、SHA256 `52332e96ef68ff8cfefd1d8d7b8c5d7b5333faa3cfac87ed4cc7b5ec3d5821c0` です。

PO の直接 NeMo API 証跡として、`ASRModel.restore_from(...); model.transcribe(audio=[...], batch_size=1, return_hypotheses=False, num_workers=0, verbose=False)` を使い、`/tmp/fluent_audio_asr_fixture/ja-pronunciation-practice5.ogg` から次の non-empty Japanese output が返りました。

`nemo_offline_transcribe_worker` の実装はこの直接 API 呼び出しと同一ではありません。worker は runtime signature を見て、`batch_size`、`return_hypotheses`、`verbose`、および受け付けられる場合の language key だけを `transcribe(...)` へ渡します。現行 worker が `num_workers` を渡すとは書きません。

```text
天 気 練 習 残 業 安 ん な り 電 波 宣 兵 電 米 宣 本 専 用 本 屋 三 円 単 位
```

この direct API evidence 単体では、offline / full-context local NeMo transcribe の viability だけを示します。accuracy、`nemo_rnnt_streaming` backend 成功、file-source full ROS graph 成功は証明しません。`nemo_rnnt_streaming` は引き続き finite attention context 不成立により fail closed / 未検証として扱います。

PO 検証では、commit `43fabab3` により `fa_in` の `pcm_file_reader` が actual PCM frame 数に基づく media-clock timestamp を出すようになった後、commit `5e3ae5a3` の opt-in smoke `src/system/fluent_audio_system/test/integration/test_file_ja_voice_frontend_real_asr_smoke.py` を `fluent-audio-runtime` で実行し、`1 passed in 130.86s` を確認しています。使用した経路は local Parakeet `.nemo` + `nemo_offline_transcribe` であり、NIM / Riva / gRPC / OpenAI は使っていません。この smoke が示すのは stated env 条件下の file-source full ROS graph で non-empty final ASR result が出たことです。accuracy、`TranscribeAudio` service integration、local `nemo_rnnt_streaming` 成功、NIM / Riva serving readiness、汎用 live microphone ASR は証明しません。
