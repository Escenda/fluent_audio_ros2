# FluentAudio ROS2

ROS2 ベースのリアルタイム音声処理基盤です。
ロボットや音声対話エージェントが、音を受け取り、整え、意味へ近づけ、必要な場所へ届けるための node / package 群を提供します。

`fluent_vision_ros2`（設計・開発: Takashi Otsuka / @takatronix）を出発点に、vision ではなく audio に責務を絞っています。

- Upstream: https://github.com/takatronix/fluent_vision_ros2
- This repository: https://github.com/Escenda/fluent_audio_ros2

## 概要

FluentAudio ROS2 は、単なる microphone adapter でも、ASR wrapper でも、speaker output helper でもありません。

FluentAudio が扱うのは、現実のロボットや AI が音を使うための一連の土台です。

- 音を入れる
- 音の形式を揃える
- 音量や帯域を整える
- ノイズや回り込みを補正する
- chunk / frame / window として扱う
- 発話、呼びかけ、会話の区切りを判断する
- 音声を文字や特徴量へ変える
- 生成された音を routing して出力する
- network や device の遅延、揺れ、ずれを扱う
- MCP や上位 agent が使える API として公開する

FluentAudio の中心思想は、次の一文です。

```text
全部を持つ。でも混ぜない。
```

必要な音声処理領域は持ちます。
しかし、それを一つの巨大な万能 node に押し込みません。

音を変える処理は processing node として見える場所に置きます。
音を判断する処理は AI node として見える場所に置きます。
音を届ける処理は IO / routing node として見える場所に置きます。
時間や network の揺れは streaming node として見える場所に置きます。
外部 device、model runtime、worker、API は backend 境界へ閉じ込めます。

たとえば microphone input を ASR に入れる pipeline は、次のように分解されます。

```text
fa_in
  -> fa_dc_offset_removal
  -> fa_high_pass
  -> fa_denoise
  -> fa_agc
  -> fa_sample_format
  -> fa_resample
  -> fa_frame_buffer
  -> fa_vad
  -> fa_asr
```

この pipeline は、単なる node の列ではありません。
音がどこから来て、どこで変わり、どこで意味を持ち、どこで拒否され得るのかを示す履歴です。

## README の位置づけ

この README は入口の地図です。
README が存在することは、package や node の完成を意味しません。

実装状態、公開 topic、公開 service、client、MCP tool、公開 config parameter の棚卸しは、次の資料を正とします。

- [docs/ノード実装状況とAPI一覧.md](docs/ノード実装状況とAPI一覧.md)

この一覧は source/config inventory です。
full build、full test、実 ROS graph launch、実デバイス、実モデル、親 VLAbor profile integration の完了証明ではありません。

現時点の棚卸しでは、leaf entry は 106 件です。
内訳は runtime node package 75 件、インターフェース package 1 件、基盤 package 2 件、計画/未実装 placeholder 28 件です。

## パッケージ構成

### IO (`src/io/`)

外部の音声 source / sink と FluentAudio graph の境界です。

- **fa_in**: microphone、file、network などの source backend から音を取り、`AudioFrame` として publish する入口
- **fa_out**: speaker、file、network などの sink backend へ `AudioFrame` を出力する出口
- **fa_record**: audio stream を保存する utility
- **fa_stream**: audio stream を外部へ配信する utility
- **fa_file_in / fa_network_in / fa_file_out / fa_network_out**: file / network I/O の planned package

`fa_in` は resample、format conversion、gain、denoise、VAD、KWS、ASR を隠しません。
`fa_out` は mixer、limiter、TTS、routing、barge-in control を隠しません。
入口と出口は大事ですが、主役ではありません。

### Processing (`src/processing/`)

音を変える処理を、責務ごとに分けて置く領域です。

- **format**: resample、sample format、bit depth、channel convert、interleave、encode/decode
- **dynamics**: gain、normalize、compressor、limiter、expander、noise gate、AGC
- **frequency**: high-pass、low-pass、band-pass、notch、EQ、de-esser
- **correction / noise**: DC offset removal、AEC、denoise、declick、hum removal
- **temporal**: trim、silence removal、delay、echo、reverb、fade、crossfade、windowing
- **spatial / channel**: pan、downmix、upmix、stereo widening、beamforming
- **analysis / feature**: STFT、log-mel、MFCC、CQT、loudness、pitch、onset、tempo
- **generation / transformation**: TTS、speech enhancement、voice conversion、speech separation など
- **routing / mixing**: mix、bus router、patchbay、loopback、ducking、sidechain、monitor mix

format conversion や denoise を AI node の中へ隠しません。
必要な変換は、pipeline 上に明示された processing node として配置します。

### Streaming (`src/streaming/`)

リアルタイム音声処理を成立させる領域です。
音質そのものよりも、途切れない、ずれない、遅れすぎないための処理を扱います。

- **fa_frame_buffer**: frame / chunk buffering
- **fa_audio_window**: audio window export / archive service
- **fa_jitter_buffer**: network jitter absorption
- **fa_clock_drift**: device / stream clock drift correction
- **fa_packet_loss_concealment**: packet loss concealment
- **fa_latency_compensation**: latency compensation
- **fa_time_alignment**: multi-source time alignment
- **fa_chunk_overlap / fa_overlap_add**: frame overlap / overlap-add

### AI (`src/ai/`)

音を意味へ近づける model node の領域です。

- **fa_vad**: voice activity detection
- **fa_kws**: keyword spotting
- **fa_asr**: speech recognition
- **fa_turn_detector**: turn end detection
- **fa_audio_embedding**: audio embedding
- **fa_sed / fa_speaker**: sound event detection / speaker 系の planned package

AI node は、model / backend を呼び、結果を publish することに集中します。
AI node 内で resample、downmix、bit depth conversion、sample format conversion を暗黙に行いません。
必要な input contract に合わない frame は、明示的に reject します。

### Apps (`src/apps/`)

audio node を使う application layer です。

- **fa_dialogue**: wake word、ASR result、turn end を `TurnContext` にまとめる node
- **fa_voice_command_router**: 音声コマンドによる start / stop / mode control
- **fa_audio_mcp**: audio window export / archive / transcription を MCP tool として公開する adapter
- **fa_safety_policy**: safety policy application の planned package

### Interfaces / System

- **fa_interfaces** (`src/interfaces/fa_interfaces`): FluentAudio の msg / srv 定義
- **fluent_audio_system** (`src/system/fluent_audio_system`): system YAML を ROS2 launch graph に展開する基盤 package

## 主要機能

### Audio edge

- source backend / sink backend を明示する
- device / file / network を ROS2 graph へ接続する
- source_id、stream_id、sample rate、channel count などの metadata を保持する
- device や backend が使えない場合、曖昧な fallback で継続しない

### DSP pipeline

- format conversion と waveform processing を分離する
- 音量、帯域、ノイズ、時間、空間、routing を node 単位で構成する
- VAD / KWS / ASR / Turn Detector の前段に必要な処理を pipeline と profile で明示する

### Voice AI

- VAD / KWS / ASR / Turn Detector / embedding を独立 node として扱う
- local model、external worker、cloud API などを backend として明示する
- missing model、missing executable、missing credential、unsupported provider を暗黙 fallback で成功扱いしない

### Dialogue / Agent tools

- wake word、ASR result、turn end を会話 context にまとめる
- audio window を export / archive / transcription できる形で MCP tool に公開する
- voice command routing を application layer として扱う

### Streaming / synchronization

- buffering、jitter、clock drift、latency、time alignment を専用 node として扱う
- network / remote operation / audio-video synchronization で起きるずれを、source / sink の中へ隠さない

## 設計原則

### Capability Contract

node / backend は、受け取れるものと受け取れないものを先に明示します。

代表的な contract は次です。

- encoding
- bit depth
- sample rate
- channel count
- layout
- normalized range
- source_id
- stream_id
- model
- provider
- runtime
- output schema

受け取れない入力を受け取った場合、勝手に変換して続けません。
変換が必要なら、`fa_sample_format`、`fa_resample`、`fa_channel_convert` などを pipeline に明示します。

### Fail Closed

壊れた状態、意味が壊れた状態、必須 resource がない状態で、処理を成功したように見せません。

禁止する例:

- missing device を別 device で代用する
- missing model を package default model で代用する
- unsupported backend を別 backend に差し替える
- ASR failure を empty success として返す
- KWS failure を no detection として返す
- VAD failure を silence として返す
- invalid format を AI node 内で変換して処理する
- stale state を current state として扱う

失敗は、失敗として見える形で扱います。

### Backend boundary

backend は ROS から切り離された実処理の境界です。
device、file、network、DSP engine、model runtime、external API、worker、process、container などを backend に閉じ込めます。

backend は ROS topic を知りません。
ROS node は parameter、topic、service、message conversion、lifecycle を担当します。

この分離により、backend は unit test しやすくなり、ROS node は graph test しやすくなります。

### Test as Proof

FluentAudio で、テストは存在確認ではありません。
テストは node の性質を証明するためにあります。

良いテストは、supported input を通し、unsupported input を拒み、startup failure、frame rejection、runtime failure、backend contract、ROS graph behavior を確認します。

Markdown、README、`package.xml`、source import 文字列を読むだけのテストは、node の性質を証明しません。

## インストール

### 前提条件

- ROS 2 Humble / Jazzy など
- ALSA: `libasound2-dev`
- audio processing backend が要求する native library
- AI backend が要求する model、external command、runtime、credential
- MCP adapter を使う場合は MCP Python SDK
- `fa_stream` で外部配信する場合は `ffmpeg`

### ビルド

```bash
colcon build --symlink-install
source install/setup.bash
```

この README は build 済みを意味しません。
現在の環境で利用する前に、対象 workspace で build / launch / graph / device / backend の代表検証を行ってください。

## 使用方法

### 個別 node 起動

各 package は原則として `node_name` と `config_file` を launch 引数に取ります。

```bash
ros2 launch fa_in fa_in.launch.py \
  node_name:=fa_in \
  config_file:=/path/to/fa_in.yaml

ros2 launch fa_vad fa_vad.launch.py \
  node_name:=fa_vad \
  config_file:=/path/to/fa_vad.yaml
```

### system YAML から起動

複数 node の graph は `fluent_audio_system` から起動します。
profile は pipeline の設計図です。

```bash
ros2 launch fluent_audio_system run.py \
  config:=/path/to/so101_kws_frontend.yaml \
  fa_in_enabled:=true \
  fa_out_enabled:=false \
  fa_in_source_id:=hw:CARD=Mic,DEV=0 \
  fa_out_sink_id:=disabled
```

代表的な KWS frontend は次の流れを構成します。

```text
fa_in
  -> fa_sample_format
  -> fa_resample
  -> fa_vad
  -> fa_kws
```

ASR、Turn Detector、dialogue context まで含める profile では、ASR / Turn Detector backend contract と `fa_dialogue` の topic contract を明示します。

### TTS を出力 pipeline に流す

```bash
ros2 launch fa_tts fa_tts.launch.py \
  node_name:=fa_tts \
  config_file:=/path/to/fa_tts.yaml

ros2 launch fa_mix fa_mix.launch.py \
  node_name:=fa_mix \
  config_file:=/path/to/fa_mix.yaml

ros2 launch fa_out fa_out.launch.py \
  node_name:=fa_out \
  config_file:=/path/to/fa_out.yaml
```

`fa_tts` は `audio/tts/frame` を出力します。
speaker device への再生は、routing / mix 後の `AudioFrame` を `fa_out` が受け取って行います。

## トピック構成

ここでは代表的な topic / service だけを示します。
完全な公開 API は [docs/ノード実装状況とAPI一覧.md](docs/ノード実装状況とAPI一覧.md) を参照してください。

### Audio stream

- `audio/frame`
- `audio/resample16k/mic`
- `audio/output/frame`
- `audio/tts/frame`

### Voice AI

- `audio/vad`
- `audio/vad/probability`
- `voice/vad_state`
- `voice/wake_word`
- `voice/asr/result`
- `voice/turn_end`
- `conversation/turn_context`

### Services

- `list_devices`
- `switch_device`
- `record`
- `speak`
- `transcribe_audio`
- `export_audio_window`
- `archive_audio_window`
- `fa_out/playback_control`

## 設定

### Node config

各 node は package 配下の `config/default.yaml` を基準に、topic、QoS、backend、model、workspace、timeout などを明示します。

backend を使う node では、必要な command、model path、provider、endpoint、credential env を config で明示します。
必須値が欠けている場合は、暗黙 fallback ではなく起動時または処理時に明確な error として扱います。

### System config

`fluent_audio_system` は system YAML を読み込み、enabled group / node を ROS2 launch action に展開します。

関連資料:

- `src/system/fluent_audio_system/config/fluent_audio_system.sample.yaml`
- `src/system/fluent_audio_system/config/profiles/`
- `src/system/fluent_audio_system/docs/仕様書.md`

## 開発・デバッグ

### ノード状態確認

```bash
ros2 node list
ros2 topic list
ros2 service list
```

### Topic 確認

```bash
ros2 topic echo /voice/vad_state
ros2 topic echo /voice/wake_word
ros2 topic echo /voice/asr/result
```

### API 棚卸し確認

```bash
find src -name package.xml | sort
```

node/package の現状確認には [docs/ノード実装状況とAPI一覧.md](docs/ノード実装状況とAPI一覧.md) を使います。

## ドキュメント

- [ENGINEERING_PHILOSOPHY.md](ENGINEERING_PHILOSOPHY.md): FluentAudio の設計・実装・テストの美学
- [PRODUCT_OWNER_ROLE.md](PRODUCT_OWNER_ROLE.md): Product Owner / 統合レビュアーとしての Codex の役割
- [NODE_ENGINEER_ROLE.md](NODE_ENGINEER_ROLE.md): node / backend 実装担当者の役割
- [CLAUDECODE_DOCUMENTATION_ROLE.md](CLAUDECODE_DOCUMENTATION_ROLE.md): 書類記載担当 ClaudeCode の役割
- [FUTURE_CODEX_MESSAGE.md](FUTURE_CODEX_MESSAGE.md): Context Compact 後に戻るためのメッセージ
- [docs/ノード実装状況とAPI一覧.md](docs/ノード実装状況とAPI一覧.md): node/package の実装状況と公開API一覧
- [docs/仕様書.md](docs/仕様書.md): repository 全体の責務境界
- [docs/アルゴリズム詳細説明書.md](docs/アルゴリズム詳細説明書.md): package 横断の処理分類と backend 境界
- [docs/テスト設計.md](docs/テスト設計.md): package 横断のテスト方針
- [docs/fa_audio_system.md](docs/fa_audio_system.md): 全体像・データフロー
- [docs/fa_audio_design.md](docs/fa_audio_design.md): 設計メモ
- 各 package 配下の `README.md` と `docs/`

## ライセンス

この repository には MIT / Apache-2.0 など複数ライセンスの package が含まれます。
各 package の `package.xml` の `<license>` を参照してください。

## 作者

Escenda + contributors

Original project: Takashi Otsuka (@takatronix) / FluentVision ROS2

## 貢献

Issue や pull request では、対象 node、入力 topic、出力 topic、使用 backend、再現手順、検証結果を明示してください。
未検証のものを検証済みとして扱わず、壊れた状態を検出できる情報を添えてください。
