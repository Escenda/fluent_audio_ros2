# FluentAudio ノード設計メモ

upstream の `fluent_vision_ros2` の設計思想（ノード分割、低遅延、YAML/launch 運用）を踏襲し、ROS2上で音声入出力・音声処理を扱うためのパッケージ群を整理します。

- 本リポジトリは音声にフォーカスしており、vision 系の実装は upstream を参照してください。
- クラウドは使わず、オフラインで完結する前提です。

## パッケージ配置（このリポジトリ）
- `src/interfaces/`: msg/srv/action（`fa_interfaces`）
- `src/io/sources/`: 入力 source adapter（`fa_in`）
- `src/io/sinks/`: 出力 sink adapter（`fa_out`）
- `src/io/utilities/`: 収録 / 配信 utility（`fa_record`, `fa_stream`）
- `src/processing/format/`: フォーマット変換（`fa_resample`）
- `src/processing/correction/`: ノイズ補正/AEC（`fa_denoise`, `fa_aec_linear`, `fa_aec_nn`）
- `src/processing/routing/`: ルーティング/ミキシング（`fa_mix`）
- `src/processing/analysis/`: 非 AI の特徴量抽出（`fa_log_mel` など）
- `src/processing/generation/`: 生成/変換（`fa_tts`）
- `src/ai/`: 音声 AI（`fa_vad`, `fa_kws`, `fa_asr`, `fa_turn_detector`, `fa_audio_embedding`, `fa_sed`, `fa_speaker`）
- `src/streaming/`: リアルタイム伝送安定化（buffer、jitter、drift、PLC、latency、alignment、overlap）
- `src/apps/voice_command/`: 音声コマンド routing（`fa_voice_command_router`）
- `src/apps/dialogue/`: 会話 orchestration（`fa_dialogue`）
- `src/apps/safety/`: 安全 policy（`fa_safety_policy`）

## 1. 優先ユースケース（MVP）
- **起動/停止/モード切替**: 音声入力で状態（例: `standby`/`command`/`dictation`/`mute`）を切り替える（`fa_voice_command_router`）。
- **マイク入力配信**: `fa_in`が`audio/frame`をPublishする。
- **VAD**: `fa_vad`が`audio/vad`をPublishする。
- **KWS**: `fa_kws`が`voice/wake_word`をPublishする。
- **ASR**: `fa_asr`がローカルASR実行ファイルを呼び、`voice/asr/result`をPublishする。
- **Turn Detection**: `fa_turn_detector`が`voice/turn_end`をPublishする。
- **TTS**: `fa_tts`が`speak`サービスで音声を生成し、`audio/tts/frame`へPublishする。再生への routing は `fa_mix` が担当する。
- **録音（WAV）**: `fa_record`が`record`サービスで`audio/frame`をWAVへ保存する。
- **配信**: `fa_stream`が`audio/frame`を外部へ送る（`ffmpeg`利用、endpoint は明示設定）。

## 2. 構成概要
```
Mic (ALSA)
  │
  ▼
fa_in_node
  ├ Pub: audio/frame (fa_interfaces/msg/AudioFrame)
  └ Srv: list_devices, switch_device
      │
      ├──────────────► fa_vad_node ── Pub: audio/vad / voice/vad_state
      ├──────────────► fa_kws ─────── Pub: voice/wake_word
      ├──────────────► fa_asr ─────── Pub: voice/asr/result
      ├──────────────► fa_turn_detector ── Pub: voice/turn_end
      ├──────────────► fa_record   ── Srv: record (WAV保存)
      └──────────────► fa_stream ── Icecast等へ送出

Text
  │
  ▼
fa_tts
  ├ Srv: speak (fa_interfaces/srv/Speak)
  └ Pub: audio/tts/frame
      │
      ▼
   fa_mix
      └ Pub: audio/output/frame
          │
          ▼
       fa_out
          └ Sub: audio/output/frame
```

## 3. ROSインターフェース（抜粋）

### 3.1 トピック
- `audio/frame`（`fa_interfaces/msg/AudioFrame`）
- `audio/vad`（`std_msgs/msg/Bool`）
- `voice/vad_state`（`fa_interfaces/msg/VadState`: VAD probability / start / end と判定元 `source_id` / `stream_id`）
- `voice/wake_word`（`fa_interfaces/msg/WakeWordResult`）
- `voice/asr/result`（`fa_interfaces/msg/AsrResult`）
- `voice/turn_end`（`fa_interfaces/msg/TurnEnd`）
- `conversation/turn_context`（`fa_interfaces/msg/TurnContext`）
- `audio/output/frame`（`fa_interfaces/msg/AudioFrame`）
- `audio/tts/frame`（`fa_interfaces/msg/AudioFrame`）

### 3.2 サービス
- `list_devices`（`fa_interfaces/srv/ListDevices`）: `fa_in`が提供
- `switch_device`（`fa_interfaces/srv/SwitchDevice`）: `fa_in`が提供
- `record`（`fa_interfaces/srv/Record`）: `fa_record`が提供
- `speak`（`fa_interfaces/srv/Speak`）: `fa_tts`が提供

## 4. パラメータ設計
原則として各パッケージ配下の `config/*.yaml` を利用します（upstream と同様に launch で差し替え可能）。

例:
- `fa_in`: `src/io/sources/fa_in/config/default.yaml`（root: `fa_in_node`）
- `fa_vad`: `src/ai/fa_vad/config/default.yaml`（root: `fa_vad_node`）
- `fa_out`: `src/io/sinks/fa_out/config/default.yaml`（root: `fa_out`）
- `fa_tts`: `src/processing/generation/fa_tts/config/default.yaml`（root: `fa_tts`）
- `fa_kws`: `src/ai/fa_kws/config/default.yaml`（root: `fa_kws`）
- `fa_asr`: `src/ai/fa_asr/config/default.yaml`（root: `fa_asr`）
- `fa_turn_detector`: `src/ai/fa_turn_detector/config/default.yaml`（root: `fa_turn_detector`）
- `fa_record`: `src/io/utilities/fa_record/config/default.yaml`（root: `fa_record`）
- `fa_stream`: `src/io/utilities/fa_stream/config/default.yaml`（root: `fa_stream`）

`fa_in`（例）:
| パラメータ | 例 | 説明 |
| --- | --- | --- |
| `audio.device_selector.mode` | `index / name` | デバイス選択モード |
| `audio.device_selector.identifier` | `"hw:CARD=Device,DEV=0"` | `name`モード時の raw ALSA hardware source |
| `audio.device_selector.index` | `1` | `index`モード時の番号 |
| `audio.sample_rate` | `48000` | Hz |
| `audio.channels` | `1 or 2` | モノラル/ステレオ |
| `audio.bit_depth` | `16 / 32` | PCMビット深度 |
| `audio.chunk_ms` | `20` | 1フレームの長さ(ms) |
| `audio.encoding` | `PCM16LE / PCM32LE / FLOAT32LE` | `AudioFrame`の格納形式 |
| `diagnostics.publish_period_ms` | `1000` | diagnostics周期(ms) |

VAD/KWS/ASR/TD を同じ音声 stream で連携する場合、`fa_vad.input_topic`、`fa_kws.audio_topic`、`fa_turn_detector.audio_topic`、`fa_asr.expected_stream_id` は同じ `AudioFrame.stream_id` に揃える必要があります。`VadState.stream_id` は `fa_vad` の入力 `AudioFrame.stream_id` を引き継ぎ、後段 node は不一致の VAD state を処理に使いません。

## 5. ランチ例
```bash
ros2 launch fa_in fa_in.launch.py
ros2 launch fa_vad fa_vad.launch.py
ros2 launch fa_out fa_out.launch.py
ros2 launch fa_tts fa_tts.launch.py
ros2 launch fa_kws fa_kws.launch.py
ros2 launch fa_asr fa_asr.launch.py
ros2 launch fa_turn_detector fa_turn_detector.launch.py
```

## 6. 実装フェーズ案（更新用メモ）
1. `fa_interfaces`: `AudioFrame`/主要srvの確定
2. `fa_in` + `fa_out`: 入出力と運用（デバイス列挙/切替、diagnostics）
3. `fa_vad`: しきい値/ヒステリシスの安定化
4. `fa_record`: WAV保存と運用（ディレクトリ/ファイル命名）
5. `src/apps/voice_command/fa_voice_command_router`: 起動/停止/モード切替（オフライン前提）
6. `src/apps/dialogue/fa_dialogue`: `voice/wake_word` / `voice/asr/result` / `voice/turn_end` を合流し、LLM/TTS/ロボット操作へ接続する会話アプリ層を実装
