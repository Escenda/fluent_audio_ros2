# FluentAudio ノード設計メモ

upstream の `fluent_vision_ros2` の設計思想（ノード分割、低遅延、YAML/launch 運用）を踏襲し、ROS2上で音声入出力・音声処理を扱うためのパッケージ群を整理します。

- 本リポジトリは音声にフォーカスしており、vision 系の実装は upstream を参照してください。
- クラウドは使わず、オフラインで完結する前提です。

## パッケージ配置（このリポジトリ）
- `src/interfaces/`: msg/srv/action（`fa_interfaces`）
- `src/io/`: 入出力・収録・配信（`fa_capture`, `fa_output`, `fa_record`, `fa_stream`）
- `src/dsp/`: DSP（`fa_ns/`, `fa_aec/`, `fa_resample/`, `fa_mix/`）
- `src/features/`: 特徴量（`fa_log_mel/`, `fa_embedding/`）
- `src/ai/`: 音声AI（`fa_vad`, `fa_tts`、将来的に `fa_kws/`, `fa_asr/`, `fa_sed/`, `fa_speaker/`）
- `src/apps/`: アプリ層（`fa_voice_command_router/`, `fa_dialogue/`, `fa_safety_policy/`）

## 1. 優先ユースケース（MVP）
- **起動/停止/モード切替**: 音声入力で状態（例: `standby`/`command`/`dictation`/`mute`）を切り替える（`fa_voice_command_router`）。
- **マイク入力配信**: `fa_capture`が`audio/frame`をPublishする。
- **VAD**: `fa_vad`が`audio/vad`をPublishする。
- **TTS**: `fa_tts`が`speak`サービスで音声を生成し、必要に応じて再生トピックへPublishする。
- **録音（WAV）**: `fa_record`が`record`サービスで`audio/frame`をWAVへ保存する。
- **配信（サンプル）**: `fa_stream`が`audio/frame`を外部へ送る（`ffmpeg`利用、用途に応じて置換）。

## 2. 構成概要
```
Mic (ALSA)
  │
  ▼
fa_capture_node
  ├ Pub: audio/frame (fa_interfaces/msg/AudioFrame)
  ├ Pub: audio/levels (std_msgs/msg/Float32MultiArray)
  └ Srv: list_devices, switch_device
      │
      ├──────────────► fa_vad_node ── Pub: audio/vad (std_msgs/msg/Bool)
      ├──────────────► fa_record   ── Srv: record (WAV保存)
      └──────────────► fa_radio_streamer (sample) ── Icecast等へ送出

Text
  │
  ▼
fa_tts
  ├ Srv: speak (fa_interfaces/srv/Speak)
  ├ Pub: audio/tts/frame (任意)
  └ Pub: audio/output/frame (play=true の場合)
                          │
                          ▼
                       fa_output
                         ├ Sub: audio/output/frame
                         └ Srv: audio/output/play_file
```

## 3. ROSインターフェース（抜粋）

### 3.1 トピック
- `audio/frame`（`fa_interfaces/msg/AudioFrame`）
- `audio/levels`（`std_msgs/msg/Float32MultiArray`）
- `audio/vad`（`std_msgs/msg/Bool`）
- `voice/vad_state`（`fa_interfaces/msg/VadState`）
- `audio/output/frame`（`fa_interfaces/msg/AudioFrame`）
- `audio/tts/frame`（`fa_interfaces/msg/AudioFrame`）

### 3.2 サービス
- `list_devices`（`fa_interfaces/srv/ListDevices`）: `fa_capture`が提供
- `switch_device`（`fa_interfaces/srv/SwitchDevice`）: `fa_capture`が提供
- `record`（`fa_interfaces/srv/Record`）: `fa_record`が提供
- `audio/output/play_file`（`fa_interfaces/srv/PlayFile`）: `fa_output`が提供
- `speak`（`fa_interfaces/srv/Speak`）: `fa_tts`が提供

## 4. パラメータ設計
原則として各パッケージ配下の `config/*.yaml` を利用します（upstream と同様に launch で差し替え可能）。

例:
- `fa_capture`: `src/io/fa_capture/config/default_audio.yaml`（root: `fa_capture_node`）
- `fa_vad`: `src/ai/fa_vad/config/default_vad.yaml`（root: `fa_vad_node`）
- `fa_output`: `src/io/fa_output/config/default.yaml`（root: `fa_output`）
- `fa_tts`: `src/ai/fa_tts/config/default.yaml`（root: `fa_tts`）
- `fa_record`: `src/io/fa_record/config/default.yaml`（root: `fa_record`）
- `fa_stream`: `src/io/fa_stream/config/default.yaml`（root: `fa_radio_streamer`）

`fa_capture`（例）:
| パラメータ | 例 | 説明 |
| --- | --- | --- |
| `audio.device_selector.mode` | `auto / index / name` | デバイス選択モード |
| `audio.device_selector.identifier` | `"USB Audio Device"` | `name`モード時の識別子 |
| `audio.device_selector.index` | `1` | `index`モード時の番号 |
| `audio.sample_rate` | `48000` | Hz |
| `audio.channels` | `1 or 2` | モノラル/ステレオ |
| `audio.bit_depth` | `16 / 32` | PCMビット深度 |
| `audio.chunk_ms` | `20` | 1フレームの長さ(ms) |
| `audio.encoding` | `pcm16 / pcm32 / float32` | `AudioFrame`の格納形式 |
| `pipeline.gain_db` | `0.0` | 入力ゲイン(dB) |
| `diagnostics.publish_period_ms` | `1000` | diagnostics周期(ms) |

## 5. ランチ例
```bash
ros2 launch fa_capture fa_capture.launch.py
ros2 launch fa_vad fa_vad.launch.py
ros2 launch fa_output fa_output.launch.py
ros2 launch fa_tts fa_tts.launch.py
```

## 6. 実装フェーズ案（更新用メモ）
1. `fa_interfaces`: `AudioFrame`/主要srvの確定
2. `fa_capture` + `fa_output`: 入出力と運用（デバイス列挙/切替、diagnostics）
3. `fa_vad`: しきい値/ヒステリシスの安定化
4. `fa_record`: WAV保存と運用（ディレクトリ/ファイル命名）
5. `src/apps/fa_voice_command_router`: 起動/停止/モード切替（オフライン前提）
6. `src/ai/fa_kws`/`src/ai/fa_asr`: コマンド入力系を拡充（オフライン前提）
