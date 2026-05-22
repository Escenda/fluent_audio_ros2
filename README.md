# FluentAudio ROS2

ロボットと AI 向けの ROS 2 音声処理基盤。

## 概要

FluentAudio ROS2 は、音声 source / sink、DSP、解析、生成、routing、streaming を責務別の ROS 2 node として扱うための package 群です。

`fa_in` / `fa_out` は device / file / network の入口と出口です。音声の加工や認識を隠して持たせず、必要な処理は pipeline 上の専用 node として明示します。

設計思想は [`ENGINEERING_PHILOSOPHY.md`](ENGINEERING_PHILOSOPHY.md) を参照してください。

## パッケージ構成

### Interfaces (`src/interfaces/`)

- **fa_interfaces**: `AudioFrame`、`EncodedAudioChunk`、feature frame、window / clip 参照、playback / recording / device service などの共通 msg / srv 定義。

### IO (`src/io/`)

- **fa_in**: source adapter。`AudioFrame` の入口。
- **fa_out**: sink adapter。`AudioFrame` の出口。
- **fa_record**: `AudioFrame` の録音サービス。
- **fa_stream**: `AudioFrame` を外部 streaming endpoint へ転送。

### Processing (`src/processing/`)

- **format**: sample format、resample、bit depth、channel、layout、encode / decode。
- **dynamics**: gain、normalize、compressor、limiter、expander、noise gate、AGC。
- **frequency**: EQ、high-pass、low-pass、band-pass、notch、de-esser。
- **temporal**: trim、silence removal、delay、echo、reverb、fade、crossfade、window。
- **correction / noise**: denoise、AEC、declick、hum、DC offset removal。
- **spatial / channel**: pan、downmix、upmix、stereo widening、beamforming。
- **analysis / feature extraction**: STFT、Log-Mel、MFCC、CQT、Pitch、Onset、Tempo、Loudness。
- **generation / transformation**: TTS など。
- **routing / mixing**: mix、bus router、sidechain、ducking、monitor mix、loopback、patchbay。

### Streaming / Synchronization (`src/streaming/`)

- **fa_frame_buffer**: frame サイズの整形。
- **fa_audio_window**: 時間範囲指定での `AudioFrame` 切り出し。
- **fa_chunk_overlap / fa_overlap_add / fa_jitter_buffer / fa_packet_loss_concealment / fa_latency_compensation / fa_clock_drift / fa_time_alignment**: streaming の安定性と同期を扱う node 群。

### AI (`src/ai/`)

- **fa_kws**: Keyword Spotting。
- **fa_turn_detector**: ターン終端検出。
- **fa_audio_embedding**: 音声 embedding 出力。
- **fa_sed / fa_speaker**: 計画段階。

### Apps (`src/apps/`)

- **fa_dialogue**: wake word / turn end から `TurnContext` を扱う app 層。
- **fa_voice_command_router**: 音声コマンドのモード制御と TTS 呼び出し。
- **fa_audio_mcp**: audio window export / archive を Agent-facing tool として公開する adapter。

### System (`src/system/`)

- **fluent_audio_system**: system YAML から launch graph を組み立てる package。

## 基本方針

- 一つの node に複数の責務を混ぜない。
- format conversion、DSP、解析、生成、streaming、apps を pipeline 上で明示する。
- `source_id` / `stream_id` を topic 名の代用にしない。
- 対応していない入力を暗黙変換せず、明示的に拒否する。
- 実装済み、未実装、未検証を混同しない。

## ビルド

```bash
cd /path/to/daihen-physical-ai/ros2_ws
colcon build --symlink-install --packages-up-to fa_interfaces
colcon build --symlink-install
source install/setup.bash
```

特定 package のみ:

```bash
colcon build --symlink-install --packages-select fa_kws fa_turn_detector fa_audio_window
```

## Pipeline 起動

複数 node の pipeline は `fluent_audio_system` で組み立てます。

```bash
ros2 launch fluent_audio_system run.py config:=<system-yaml>
```

## 開発・デバッグ

```bash
ros2 topic list
ros2 topic echo /audio/frame
ros2 node list
```

多くの processing node は diagnostics を publish し、入出力 contract や reject 状況を出力します。

## ドキュメント

- [`ENGINEERING_PHILOSOPHY.md`](ENGINEERING_PHILOSOPHY.md) — 設計・実装・テストの思想。
- [`PRODUCT_OWNER_ROLE.md`](PRODUCT_OWNER_ROLE.md) — Codex の役割定義。
- [`NODE_ENGINEER_ROLE.md`](NODE_ENGINEER_ROLE.md) — node 実装担当者の役割定義。
- [`CLAUDECODE_DOCUMENTATION_ROLE.md`](CLAUDECODE_DOCUMENTATION_ROLE.md) — 書類記載担当者の役割定義。
- [`FUTURE_CODEX_MESSAGE.md`](FUTURE_CODEX_MESSAGE.md) — Context Compact 後の復帰用メッセージ。

## ライセンス

親リポジトリの方針に従います。
