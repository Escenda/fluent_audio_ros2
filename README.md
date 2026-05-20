# FluentAudio ROS2

ROS2 ベースのリアルタイム音声処理システムです。ロボットや音声対話アプリケーションで使う audio input、audio output、DSP、streaming、VAD/KWS/ASR/Turn Detector、MCP 連携を、責務の分かれた package / node として提供します。

`fluent_vision_ros2`（設計・開発: Takashi Otsuka / @takatronix）を出発点に、vision ではなく audio に責務を絞っています。

- Upstream: https://github.com/takatronix/fluent_vision_ros2
- This repository: https://github.com/Escenda/fluent_audio_ros2

## 概要

FluentAudio ROS2 は、音声入力から音声AI、音声出力、外部 agent 連携までを ROS2 graph として構成するための repository です。

単一の巨大な万能 node ではなく、format conversion、dynamics、frequency、correction、temporal、spatial、analysis、generation、routing、streaming を、それぞれ責務のはっきりした node として分割します。

たとえば microphone input を ASR に入れる pipeline は、次のように構成できます。

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

この分割により、どこで音が変わったのか、どの node がどの topic / service を持つのか、どこで失敗したのかを追跡できます。

## パッケージ構成

完全な node/package 一覧、実装状態、topic、service、client、MCP tool、公開 config parameter は [docs/ノード実装状況とAPI一覧.md](docs/ノード実装状況とAPI一覧.md) を参照してください。

この一覧は source/config inventory です。full build、full test、実 ROS graph launch、実デバイス、実モデル、親 VLAbor profile integration の完了証明ではありません。

現時点の棚卸しでは、leaf entry は 106 件です。内訳は runtime node package 75 件、インターフェース package 1 件、基盤 package 2 件、計画/未実装 placeholder 28 件です。

### IO (`src/io/`)

- **fa_in**: 明示 backend で選択した入力源から `AudioFrame` を publish する source node
- **fa_out**: 明示 backend で選択した出力先へ `AudioFrame` を出力する sink node
- **fa_record**: audio stream を WAV などへ保存する recording utility
- **fa_stream**: audio stream を外部へ配信する utility
- **fa_file_in / fa_network_in / fa_file_out / fa_network_out**: file / network I/O の planned package

### Processing (`src/processing/`)

- **format**: resample、sample format、bit depth、channel convert、interleave、encode/decode
- **dynamics**: gain、normalize、compressor、limiter、expander、noise gate、AGC
- **frequency**: high-pass、low-pass、band-pass、notch、EQ、de-esser
- **correction/noise**: DC offset removal、AEC、denoise、declick、hum removal
- **temporal**: trim、silence removal、delay、echo、reverb、fade、crossfade、windowing
- **spatial/channel**: pan、downmix、upmix、stereo widening、beamforming
- **analysis/feature**: STFT、log-mel、MFCC、CQT、loudness、pitch、onset、tempo
- **generation**: TTS、および speech enhancement / conversion / separation などの planned package
- **routing/mixing**: mix、bus router、patchbay、loopback、ducking、sidechain、monitor mix

### Streaming (`src/streaming/`)

- **fa_frame_buffer**: audio chunk / frame buffering
- **fa_audio_window**: audio window export / archive service
- **fa_jitter_buffer**: network jitter absorption
- **fa_clock_drift**: clock drift correction
- **fa_packet_loss_concealment**: packet loss concealment
- **fa_latency_compensation**: latency compensation
- **fa_time_alignment**: multi-source time alignment
- **fa_chunk_overlap / fa_overlap_add**: frame overlap / overlap-add

### AI (`src/ai/`)

- **fa_vad**: voice activity detection
- **fa_kws**: keyword spotting
- **fa_asr**: speech recognition
- **fa_turn_detector**: turn end detection
- **fa_audio_embedding**: audio embedding
- **fa_sed / fa_speaker**: sound event detection / speaker 系の planned package

### Apps (`src/apps/`)

- **fa_dialogue**: wake word、ASR、turn end を会話 context にまとめる node
- **fa_voice_command_router**: 音声コマンドによる start / stop / mode control
- **fa_audio_mcp**: audio window export / archive / transcription を MCP tool として公開する adapter
- **fa_safety_policy**: safety policy application の planned package

### Interfaces / System

- **fa_interfaces** (`src/interfaces/fa_interfaces`): FluentAudio の msg / srv 定義
- **fluent_audio_system** (`src/system/fluent_audio_system`): system YAML を ROS2 launch graph に展開する orchestration package

## 主要機能

### 音声入出力

- microphone、file、network などの入力源を backend として明示
- speaker、file、network などの出力先を backend として明示
- 入出力 device の列挙、切り替え、playback control
- diagnostics による状態公開

### DSP pipeline

- format conversion と waveform processing を分離
- volume / dynamics、frequency、noise correction、temporal、spatial、routing を node 単位で構成
- VAD、ASR、KWS などの前段に必要な resample、sample format、buffering を明示的に配置

### 音声AI

- VAD / KWS / ASR / Turn Detector を独立 node として提供
- local model、external worker、cloud API などの backend を config で明示
- backend、model、endpoint、credential が未設定の場合に暗黙 fallback で成功扱いしない

### 音声対話・agent連携

- wake word、ASR result、turn end を `TurnContext` に統合
- audio window の export / archive / transcription を MCP tool として公開
- 音声コマンド routing と TTS 連携を application layer として分離

### リアルタイム処理

- buffering、jitter、clock drift、latency、time alignment を streaming layer に分離
- network / WebRTC / remote operation で問題になりやすい遅延、欠落、同期ずれを明示的な node として扱う

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

複数 node の graph は `fluent_audio_system` から起動できます。

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

ASR、Turn Detector、dialogue context まで含めた検証用 profile は `so101_voice_frontend.yaml` を参照してください。

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

`fa_tts` は `audio/tts/frame` を出力します。speaker device への再生は、routing / mix 後の `AudioFrame` を `fa_out` が受け取って行います。

## トピック構成

ここでは代表的な topic / service だけを示します。完全な公開 API は [docs/ノード実装状況とAPI一覧.md](docs/ノード実装状況とAPI一覧.md) を参照してください。

### Audio stream

- `audio/frame` または node config の `input_topic` / `output_topic`
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

backend を使う node では、必要な command、model path、endpoint、credential env を config で明示します。必須値が欠けている場合は、暗黙 fallback ではなく起動時または処理時に明確な error として扱います。

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

- [docs/ノード実装状況とAPI一覧.md](docs/ノード実装状況とAPI一覧.md): node/package の実装状況と公開API一覧
- [docs/仕様書.md](docs/仕様書.md): repository 全体の責務境界
- [docs/アルゴリズム詳細説明書.md](docs/アルゴリズム詳細説明書.md): package 横断の処理分類と backend 境界
- [docs/テスト設計.md](docs/テスト設計.md): package 横断のテスト方針
- [docs/fa_audio_system.md](docs/fa_audio_system.md): 全体像・データフロー
- [docs/fa_audio_design.md](docs/fa_audio_design.md): 設計メモ
- 各 package 配下の `README.md` と `docs/`

## ライセンス

この repository には MIT / Apache-2.0 など複数ライセンスの package が含まれます。各 package の `package.xml` の `<license>` を参照してください。

## 作者

Escenda + contributors

Original project: Takashi Otsuka (@takatronix) / FluentVision ROS2

## 貢献

Issue や pull request では、対象 node、入力 topic、出力 topic、使用 backend、再現手順、検証結果を明示してください。暗黙 fallback や未検証の完了宣言は避け、壊れた状態を検出できる情報を添えてください。
