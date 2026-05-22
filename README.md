# FluentAudio ROS2

ロボットと AI 向けの ROS 2 音声処理基盤。

## 概要

FluentAudio ROS2 は、microphone / file / network からの音声入力、DSP 処理 (format / dynamics / frequency / temporal / correction / spatial)、解析・特徴量抽出 (STFT / Mel / MFCC ほか)、音声 AI (VAD / KWS / ASR / Turn Detector)、TTS / mixing / routing、streaming / synchronization、speaker / file / network への出力までを、責務ごとに分けた ROS 2 node 群として提供します。

各 node の実装状態と公開 API は [`docs/ノード実装状況とAPI一覧.md`](docs/ノード実装状況とAPI一覧.md) を一次情報として参照してください。設計思想は [`ENGINEERING_PHILOSOPHY.md`](ENGINEERING_PHILOSOPHY.md) を参照してください。

## パッケージ構成

### Interfaces (`src/interfaces/`)
- **fa_interfaces**: 全 node 共通の msg / srv 定義 (`AudioFrame`, `EncodedAudioChunk`, `VadState`, `AsrResult`, `TurnContext`, `TurnEnd`, `WakeWordResult`, `StftFrame`, `LogMelFrame`, `MfccFrame`, `CqtFrame`, `PitchFrame`, `OnsetFrame`, `TempoFrame`, `LoudnessFrame`, `AudioEmbeddingFrame`, `Speak`, `TranscribeAudio`, `Record`, `PlaybackControl`, `ListDevices`, `SwitchDevice`, `ExportAudioWindow`, `ArchiveAudioWindow` ほか)

### IO (`src/io/`)
- **fa_in**: source adapter (ALSA / file / network)。`AudioFrame` の入口
- **fa_out**: sink adapter (ALSA / file / network)。`AudioFrame` の出口
- **fa_record**: `AudioFrame` の録音サービス
- **fa_stream**: `AudioFrame` を外部 streaming endpoint へ転送 (ffmpeg)
- **fa_file_in / fa_file_out / fa_network_in / fa_network_out**: 計画段階

### Processing (`src/processing/`)

format conversion (`format/`)
- **fa_sample_format**: PCM ↔ FLOAT などのサンプル表現変換
- **fa_resample**: サンプリングレート変換 (mic / ref の 2 系統対応)
- **fa_bit_depth**: bit depth 変換
- **fa_channel_convert**: channel 数変換 (mono / stereo)
- **fa_interleave**: interleaved ↔ planar layout 変換
- **fa_encode / fa_decode**: 外部 codec (opus / ogg ほか) のエンコード・デコード
- **fa_format**: 計画段階

dynamics (`dynamics/`)
- **fa_gain / fa_normalize / fa_compressor / fa_limiter / fa_expander / fa_noise_gate / fa_agc**

frequency (`frequency/`)
- **fa_eq / fa_high_pass / fa_low_pass / fa_band_pass / fa_notch / fa_deesser**
- **fa_filter / fa_spectral_subtraction / fa_wiener**: 計画段階

temporal (`temporal/`)
- **fa_trim / fa_silence_removal / fa_delay / fa_echo / fa_reverb / fa_fade / fa_crossfade / fa_window**
- **fa_time_stretch / fa_pitch_shift**: 計画段階

correction / noise (`correction/`)
- **fa_denoise / fa_aec_linear / fa_aec_nn / fa_declick / fa_hum / fa_dc_offset_removal**
- **fa_declip / fa_debreath / fa_dereverb / fa_wind**: 計画段階

spatial / channel (`spatial/`)
- **fa_pan / fa_downmix / fa_upmix / fa_stereo_widening / fa_beamforming**
- **fa_source_separation / fa_binaural / fa_ambisonics**: 計画段階

analysis / feature extraction (`analysis/`)
- **fa_stft / fa_log_mel / fa_mfcc / fa_cqt / fa_pitch / fa_onset / fa_tempo / fa_loudness**

generation / transformation (`generation/`)
- **fa_tts**: Speak サービス + 音声 frame 出力
- **fa_voice_conversion / fa_speech_enhancement / fa_speech_separation / fa_speech_translation / fa_neural_codec / fa_neural_vocoder / fa_super_resolution / fa_music_source_separation**: 計画段階

routing / mixing (`routing/`)
- **fa_mix / fa_bus_router / fa_sidechain / fa_ducking / fa_monitor_mix / fa_loopback / fa_patchbay**

### Streaming / Synchronization (`src/streaming/`)
- **fa_frame_buffer**: frame サイズの整形
- **fa_chunk_overlap / fa_overlap_add**: chunk のオーバーラップ処理
- **fa_jitter_buffer / fa_packet_loss_concealment / fa_latency_compensation / fa_clock_drift / fa_time_alignment**: ネットワーク・device 揺れ補正
- **fa_audio_window**: 時間範囲指定での `AudioFrame` 切り出し (ExportAudioWindow / ArchiveAudioWindow 提供)

### AI (`src/ai/`)
- **fa_vad**: Voice Activity Detection (silero-vad worker 経由)
- **fa_kws**: Keyword Spotting (sherpa-onnx worker 経由)
- **fa_asr**: ASR + Transcribe サービス。標準 ASR backend は in-process `parakeet_multilingual_buffered` で、legacy external backend は明示選択時だけ使う
- **fa_turn_detector**: ターン終端検出 (smart-turn ONNX worker 経由)
- **fa_audio_embedding**: 音声 embedding 出力
- **fa_sed / fa_speaker**: 計画段階

### Apps (`src/apps/`)
- **fa_dialogue**: wake → ASR → turn end → TurnContext 統合
- **fa_voice_command_router**: モード切替 (standby / command / dictation / mute) と TTS 呼び出し
- **fa_audio_mcp**: MCP サーバ (`export_audio_window`, `archive_audio_window`, `transcribe_audio` ツールを ROS service 経由で公開)
- **fa_safety_policy**: 計画段階

### System (`src/system/`)
- **fluent_audio_system**: system YAML から launch graph を組み立てる。CLI `list_required_packages` 付き

> 公開 API (Subscriptions / Publishers / Services / Clients / MCP tools / config parameter) の正式一覧は [`docs/ノード実装状況とAPI一覧.md`](docs/ノード実装状況とAPI一覧.md)。inventory 記載時点で leaf entries 106、runtime node 75、interface 1、foundation 2、計画段階 28。

## 主要機能

### 音声入出力
- ALSA / file / network からの音声入力 (`fa_in`)
- ALSA / file / network への音声出力 (`fa_out`)
- 録音 (`fa_record`)、外部 streaming endpoint への転送 (`fa_stream`)
- 時間範囲指定での `AudioFrame` 切り出し / 書き出し (`fa_audio_window`)

### DSP 処理
- format conversion: sample format / bit depth / channel / layout / resample / codec encode-decode
- dynamics: gain / normalize / compressor / limiter / expander / noise gate / AGC
- frequency: EQ / high-pass / low-pass / band-pass / notch / de-esser
- temporal: trim / silence removal / delay / echo / reverb / fade / crossfade / window
- correction: denoise / AEC (linear / NN) / declick / hum / DC offset removal
- spatial: pan / downmix / upmix / stereo widening / beamforming
- routing / mixing: mix / bus / sidechain / ducking / monitor mix / loopback / patchbay

### 解析・特徴量抽出
- STFT / Log-Mel / MFCC / CQT / Pitch / Onset / Tempo / Loudness / Audio Embedding

### 音声 AI
- VAD (`fa_vad`)、Keyword Spotting (`fa_kws`)、ASR (`fa_asr`)、Turn Detector (`fa_turn_detector`)
- TTS (`fa_tts`)

### 対話・コマンド
- wake → ASR → turn end の統合 (`fa_dialogue`)
- 音声コマンドのモード制御 (`fa_voice_command_router`)

### Pipeline 組み立て
- `fluent_audio_system` の system YAML から ROS 2 launch graph に展開

### エージェント連携
- MCP サーバ (`fa_audio_mcp`) が `export_audio_window` / `archive_audio_window` / `transcribe_audio` を提供。LLM / VLM 系エージェントから ROS service 経由で呼べる

## インストール

### 前提条件
- ROS 2 (親リポジトリの ROS 2 環境に追従)
- 各 backend が必要とする外部依存 (ALSA、ONNX Runtime、sherpa-onnx、ffmpeg、外部 worker、各種 codec など)
- node が要求する model / device は backend ごとに異なる

詳細は各 package の `package.xml` と `docs/backends/<backend_name>.md` を参照。

### ビルド
```bash
cd /path/to/daihen-physical-ai/ros2_ws
colcon build --symlink-install --packages-up-to fa_interfaces
colcon build --symlink-install
source install/setup.bash
```

特定 package のみ:
```bash
colcon build --symlink-install --packages-select fa_vad fa_kws fa_asr
```

## 使用方法

### 単一 node の起動
各 node は `node_name` と `config_file` を引数に取る launch を持ちます。

```bash
ros2 launch fa_vad fa_vad.launch.py \
    node_name:=fa_vad \
    config_file:=$(ros2 pkg prefix fa_vad)/share/fa_vad/config/default.yaml
```

### Pipeline (system YAML) からの起動
複数 node の pipeline は `fluent_audio_system` で組み立てます。

```bash
ros2 launch fluent_audio_system <profile>.launch.py
```

YAML schema と展開ルールは `fluent_audio_system` の `docs/` を参照。

## トピック / サービス構成

### 音声 frame
- `audio/frame` / `fa_interfaces/msg/AudioFrame`: 基本 frame
- `audio/encoded/mic` / `fa_interfaces/msg/EncodedAudioChunk`: codec 出力

### 音声 AI
- `voice/vad_state` / `fa_interfaces/msg/VadState`
- `voice/wake_word` / `fa_interfaces/msg/WakeWordResult`
- `voice/asr/result` / `fa_interfaces/msg/AsrResult`
- `voice/turn_end` / `fa_interfaces/msg/TurnEnd`
- `conversation/turn_context` / `fa_interfaces/msg/TurnContext`

### 解析 / 特徴量
- `audio/stft/*`, `audio/log_mel/*`, `audio/mfcc/*`, `audio/cqt/*`, `audio/pitch/*`, `audio/onset/*`, `audio/tempo/*`, `audio/loudness/*`, `audio/embedding/*` (対応する `*Frame` msg)

### サービス
- `transcribe_audio` / `fa_interfaces/srv/TranscribeAudio`
- `speak` / `fa_interfaces/srv/Speak`
- `fa_out/playback_control` / `fa_interfaces/srv/PlaybackControl`
- `fa_in/list_devices` `fa_in/switch_device`
- `export_audio_window` / `archive_audio_window` (`fa_audio_window` 提供、MCP 経由でも利用)
- `fa_record/record` / `fa_interfaces/srv/Record`

### 識別子
- `source_id` / `stream_id` で識別する。topic 名による識別は代用にしない。stream_id mismatch は reject 対象

> 各 node の default topic 名、QoS、config parameter は [`docs/ノード実装状況とAPI一覧.md`](docs/ノード実装状況とAPI一覧.md) を参照。

## 設定

### 設定階層
1. 各 package の `config/default.yaml` (単独起動の最小デフォルト)
2. user / profile / system YAML (`fluent_audio_system` などが扱う)
3. launch 引数による上書き

package-local の便利 default を、profile が指定すべき値の代用にしません。

### Profile 例
- **wake-word only**: `fa_in` → format → `fa_kws`
- **voice frontend (ASR まで)**: `fa_in` → 補正・format → `fa_vad` → `fa_kws` → `fa_asr` → `fa_turn_detector` → `fa_dialogue`
- **AEC + loopback**: `fa_in` + `fa_out` (loopback) → `fa_aec_linear` → downstream
- **network streaming**: `fa_in` → `fa_encode` → `fa_stream` / 受信側で `fa_jitter_buffer` → `fa_decode`

## 開発・デバッグ

### トピック / ノード確認
```bash
ros2 topic list
ros2 topic echo /audio/frame
ros2 node info /fa_vad
```

### 診断情報
多くの processing node は `diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`) を publish し、入出力 contract や reject 状況を出力します。

### Backend 境界
- backend は ROS から切り離された実処理 (device / file / network / DSP / model runtime / 外部 worker / 外部 API)
- backend code は `rclcpp` / `rclpy` / ROS message を知らない
- ROS node 側が parameter / topic / service / lifecycle / message conversion を担当
- backend の supported capability / startup failure / frame rejection / runtime fatal は `docs/backends/<backend_name>.md` に集約

## テスト

テストはアルゴリズムと契約が壊れていないことを実行経路で証明するためのものです。

検証対象:
- backend public API が supported config を受け取り、unsupported config を拒否する
- DSP アルゴリズムの数値的性質 (resample 後の主要周波数、limiter の clipping 抑制など)
- supported / unsupported `AudioFrame` の validation 経路
- startup failure (missing model / device / unknown backend / invalid config)
- frame rejection (unsupported encoding / sample_rate mismatch / stream_id mismatch / malformed payload)
- runtime fatal shutdown、launch argument / config validation、ROS graph 上の publish / subscribe

詳細は [`docs/テスト設計.md`](docs/テスト設計.md)。

## ドキュメント

- [`docs/仕様書.md`](docs/仕様書.md) — 全体仕様
- [`docs/アルゴリズム詳細説明書.md`](docs/アルゴリズム詳細説明書.md) — アルゴリズム説明
- [`docs/テスト設計.md`](docs/テスト設計.md) — 検証性質の設計
- [`docs/ノード実装状況とAPI一覧.md`](docs/ノード実装状況とAPI一覧.md) — Node / API inventory (一次情報)
- [`docs/backends/`](docs/backends/) — backend ごとの contract / failure 条件
- [`docs/fa_audio_design.md`](docs/fa_audio_design.md), [`docs/fa_audio_system.md`](docs/fa_audio_system.md) — 設計資料
- [`docs/agent_runtime_audio_tools.md`](docs/agent_runtime_audio_tools.md), [`docs/roadmap_placeholders.md`](docs/roadmap_placeholders.md) — エージェント連携 / 未実装 placeholder
- [`ENGINEERING_PHILOSOPHY.md`](ENGINEERING_PHILOSOPHY.md) — 設計思想

役割定義: [`PRODUCT_OWNER_ROLE.md`](PRODUCT_OWNER_ROLE.md), [`NODE_ENGINEER_ROLE.md`](NODE_ENGINEER_ROLE.md), [`CLAUDECODE_DOCUMENTATION_ROLE.md`](CLAUDECODE_DOCUMENTATION_ROLE.md), [`FUTURE_CODEX_MESSAGE.md`](FUTURE_CODEX_MESSAGE.md)

## ライセンス

親リポジトリの方針に従います。
