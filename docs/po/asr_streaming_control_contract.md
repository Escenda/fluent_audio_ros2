# ASR Streaming Control Contract PO 設計資料

## 位置づけ

この資料は、FluentAudioROS2 における ASR / KWS / TurnDetector の streaming control contract と、現行標準 ASR backend `parakeet_multilingual_buffered` の control policy を整理する Product Owner 設計資料です。

これは実装完了報告ではありません。
message、service、node 実装、backend 実装、launch、profile、test、代表検証の完了をこの資料だけで意味しません。

この資料が標準として扱う ASR backend は `parakeet_multilingual_buffered` です。
`nemo_rnnt_streaming`、`nemo_offline_transcribe`、Whisper、OpenAI、NIM、Riva、gRPC、stdin/stdout JSONL worker は、残っている docs / code path がある場合でも legacy optional path であり、標準経路、default、参照実装、fallback、validation substitute ではありません。

## 背景

ASR は audio chunk を backend に投げるだけでは成立しません。
どこで stream を始め、どこまでを発話として扱い、どの条件で transcript を final にし、どの条件で backend state を破棄するかが product contract になります。

`fa_vad` は audio を ASR へ渡す node ではありません。
`fa_vad` は発話状態や発話区間を示す side signal producer です。
`fa_turn_detector` も ASR の backend state を直接操作するのではなく、turn close / commit の side signal を出します。

`fa_asr` は、自分の backend contract に合う ASR-ready audio stream だけを読み、VAD、TurnDetector、manual control、timeout を control side signal として扱います。

## 原則

### backend は control を知らない

backend は ROS topic、VAD topic、TurnDetector topic、control trigger schema を知りません。
backend が知る lifecycle は次だけです。

```text
start_stream
push_audio
finish
cancel
```

どの trigger で `start_stream` するか、どの timeout で `finish` するか、どの stream contract violation で `cancel` または fail closed するかは node / control 層の責務です。

### standard backend は full-context streaming policy を明示する

`parakeet_multilingual_buffered` は multilingual Parakeet 1.1B `.nemo` を full-context model として扱います。
現行 policy は rolling buffer / chunked re-decode です。
cache-aware streaming backend に失敗したときの fallback ではありません。

### 未対応を隠さない

audio frame の gap、overlap、reorder、sample rate change、channel change、encoding change、identity mismatch を黙って飲み込むと transcript の意味が壊れます。
壊れた stream は commit せず、cancel または fail closed に進めます。

## Control Phase

control phase は、node が外部 signal や timeout をどの意味で受け取るかを表します。

| phase | 意味 | 主な対象 |
| --- | --- | --- |
| `enable_on` | node の処理を開始または許可する条件 | ASR / KWS / TD |
| `disable_on` | node の処理を一時的に止める条件 | KWS / TD / ASR |
| `commit_on` | 現在の stream / turn / result を確定する条件 | ASR / TD |
| `cancel_on` | 現在の stream / turn / result を破棄する条件 | ASR / TD |
| `reset_on` | 内部状態を破棄して初期状態に戻す条件 | ASR / KWS / TD |

同時発火時は `reset_on`、`cancel_on`、`commit_on`、`disable_on`、`enable_on` の順で強く扱います。
壊れた stream、明示 cancel、内部状態 reset が必要な状態を transcript commit で覆ってはいけません。

## Control Trigger

trigger は単一設定ではなく、複数列挙できる配列として扱います。

```text
control.<phase>.triggers[]
```

trigger type の候補は次です。

| type | 意味 | 備考 |
| --- | --- | --- |
| `topic` | ROS topic message の field / key による trigger | VAD 専用にしない |
| `timeout` | node が保持する時刻基準による trigger | fallback ではなく明示 policy |
| `lifecycle` | node lifecycle / graph lifecycle による trigger | startup / shutdown / deactivate など |
| `stream_contract_violation` | AudioFrame stream contract 破綻による trigger | commit ではなく cancel/fail closed |
| `backend_error` | backend runtime failure による trigger | error result と fail closed を分けて設計 |
| `service_request` | service や action による明示 request | manual commit / cancel など |

`topic` は ROS transport 名、`stream_id` は payload の logical stream identity です。
この 2 つを混同してはいけません。
message type、field path、operator、value 型が曖昧な場合、文字列変換で成功扱いにせず起動時に fail closed とします。

timeout は fallback ではありません。
timeout は「この条件で stream を確定または破棄する」という明示 policy です。

## ASR Lifecycle

`fa_asr` は少なくとも次の概念状態を持ちます。

| state | 意味 |
| --- | --- |
| `idle` | backend stream が存在せず、audio を transcript 対象として扱っていない |
| `armed` | trigger 待ち、または pre-roll を保持している |
| `streaming` | backend stream が開始され、audio chunk を受け付けている |
| `committing` | final transcript を確定中 |
| `cancelling` | transcript を出さずに backend state を破棄中 |
| `failed` | node 継続が危険な失敗状態 |

概念 transition は次です。

```text
idle
  -- enable_on --> armed or streaming

armed
  -- valid audio --> streaming
  -- cancel_on/reset_on --> idle

streaming
  -- commit_on --> committing
  -- cancel_on --> cancelling
  -- stream_contract_violation fatal --> failed
  -- backend_error fatal --> failed

committing
  -- backend final ok --> idle
  -- backend final error/empty speech final --> failed or cancelling

cancelling
  -- backend cancel ok --> idle

failed
  -- lifecycle reset or node restart --> idle
```

## Partial / Final / Error

`parakeet_multilingual_buffered` は rolling buffer を chunk 境界で re-decode し、変化した text を partial hypothesis として publish できます。

- `partial`: 未確定 hypothesis。後続 update で変わり得る。会話履歴や committed transcript にしない。
- `final`: VAD / TurnDetector / timeout close の `finish()` で返る committed transcript。
- `cancelled`: stream が破棄され、transcript を採用しない。
- `error`: backend error、contract violation、config error、speech energy が十分ある empty final など。

partial を UI 表示に使うかは上位 app の責務です。
`fa_asr` は event type、stream identity、time range、reason を明示して publish します。

## Standard Backend Contract

`parakeet_multilingual_buffered` の required contract は次です。

```yaml
backend:
  name: parakeet_multilingual_buffered
  model_path: /models/asr/parakeet-1.1b-multilingual.nemo
  model: ""
  language: ""
  language_policy: auto_detect
  sample_rate_hz: 16000
  channels: 1
  chunk_size_samples: 1600
  chunk_ms: 0
  emit_partial: true
  max_buffer_sec: 30.0
  speech_energy_threshold: 0.001
```

`backend.model_path` と `backend.model` はどちらか一方だけを設定します。
model は multilingual Parakeet 1.1B を識別する必要があります。
multilingual Parakeet 1.1B ではない model は拒否します。

`backend.language` は空文字、`backend.language_policy` は `auto_detect` です。
language を profile が保証できない場合に固定値で補ってはいけません。

## AudioFrame Stream Contract

`fa_asr` が backend に渡す audio は次だけです。

| field | required |
| --- | --- |
| encoding | `FLOAT32LE` |
| sample rate | `16000` |
| channels | `1` |
| bit depth | `32` |
| samples | non-empty, finite, normalized |

`fa_asr` は sample format conversion、resample、downmix、normalize、denoise、zero fill を行いません。
必要な変換は upstream processing node で明示します。

streaming ASR / KWS / TurnDetector は、AudioFrame が連続していることを前提に state を持ちます。
後続実装では少なくとも次の情報を検証対象にします。

- `source_id`
- `stream_id`
- `epoch`
- `seq`
- `start_sample`
- `frame_count`
- `sample_rate`
- `channels`
- `encoding`
- `layout`

`AudioFrame.header.stamp` は時刻 anchor であり、sample continuity の代替ではありません。

## Legacy Optional Paths

次の path は標準 ASR path ではありません。
docs が残る場合は optional / legacy として扱います。

| path | 扱い |
| --- | --- |
| `nemo_rnnt_streaming` | legacy experimental cache-aware streaming slot。現行 Parakeet full-context standard ではない。 |
| `nemo_offline_transcribe` | legacy optional non-streaming command / worker backend。過去 evidence はあるが標準 streaming control policy ではない。 |
| Whisper / `whisper.cpp` | legacy optional external worker。標準 / fallback ではない。 |
| `parakeet_worker` / `local_command` | legacy optional external command worker。標準 / fallback ではない。 |
| OpenAI Realtime / Transcriptions | optional external API worker slot。標準 / fallback ではない。 |
| NIM / Riva / gRPC | external serving boundary。現行標準 backend の validation substitute ではない。 |

## Test Design 観点

後続テストは Markdown 文字列検査ではなく、node/backend の実行経路で性質を検証します。

- `enable_on` で `idle` から `streaming` へ進む。
- `commit_on` で `finish()` が呼ばれ、final だけが committed transcript になる。
- `cancel_on` / `reset_on` で rolling buffer と backend session が破棄される。
- partial は後続 update で変わり得る未確定 hypothesis として扱われる。
- speech energy が十分ある empty final は success にならない。
- `FLOAT32LE` / 16 kHz / mono 以外は backend state に触れる前に reject される。
- language が保証できない場合は `auto_detect` policy を要求し、固定 language で補わない。
- unsupported model、Whisper、NIM、Riva、gRPC、legacy worker へ fallback しない。
- timeout は明示 reason を持つ commit / cancel policy として扱われる。

## 未決事項

| 項目 | 未決内容 |
| --- | --- |
| 共通 control package 名 | `fa_control`、`fluent_audio_control`、既存 system package 内部実装のどれにするか |
| trigger schema の message 化 | YAML-only にするか、service/action/message でも表現するか |
| topic trigger の key path | ROS message field path の表現方法 |
| operator の型検証 | bool / int / float / string の比較規則 |
| timeout clock | ROS time / steady clock / media clock の使い分け |
| ASR pre-roll | `armed` 状態で何 ms の audio を保持するか |
| forced commit | `max_stream_duration` を commit と cancel のどちらに分類するか |
| partial event schema | `AsrEvent` / `AsrResult` の field 追加要否 |

未決事項は default や fallback で隠してはいけません。

## 後続作業

1. 共通 control schema の仕様書を作る。
2. `fa_asr` 仕様書へ streaming lifecycle / trigger contract を反映する。
3. `fa_kws` / `fa_turn_detector` 仕様書へ同じ control schema を反映する。
4. `parakeet_multilingual_buffered` backend docs と system profile contract を同期する。
5. backend capability validation の実装単位を維持する。
6. state transition / timeout / frame violation / empty speech final / no-fallback の意味あるテストを維持する。
7. representative pipeline で実音声の partial / final / cancel event を検証する。

完了判定は、この資料の存在ではなく、実装、テスト、代表検証がそろった時点で行います。
