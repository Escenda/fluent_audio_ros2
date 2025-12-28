# FluentAudio オーディオシステム資料

本資料は FluentAudio（本リポジトリ）の音声パッケージ群を、ROS2のトピック/サービス観点で俯瞰するためのメモです。未実装の将来構想（KWS/ASR/SED、アプリ層など）も含みます。

## 1. 目的（制約込み）
1. **入力/出力デバイスの分離**: マイクとスピーカーを別ノードにして、デバイス占有や遅延調整を独立管理する。
2. **低遅延・組み合わせ容易性**: upstream と同様に、疎結合（topic/srv）で差し替え可能な構成にする。
3. **オフライン前提**: クラウドを使わず、音声入力で「起動/停止/モード切替」を実現できる土台を作る。
4. **再利用可能なインターフェース**: 音声フレームは `fa_interfaces/msg/AudioFrame` に統一する。

## 2. ノード構成（実装済み）

| ノード | 役割 | 主な入出力 |
| --- | --- | --- |
| `fa_capture_node` | マイク/ライン入力を開きPCM配信 | Pub: `audio/frame`, `audio/levels` / Srv: `list_devices`, `switch_device` |
| `fa_vad_node` | 軽量VAD（RMS/ヒステリシス） | Sub: `audio/frame` → Pub: `audio/vad` |
| `fa_record` | `audio/frame` をWAVへ録音 | Sub: `audio/frame` / Srv: `record` |
| `fa_tts` | テキスト→音声合成 | Srv: `speak` →（任意）Pub: `audio/tts/frame` /（既定）Pub: `audio/output/frame` |
| `fa_output` | スピーカーを開いてPCMを再生 | Sub: `audio/output/frame` / Srv: `audio/output/play_file` |
| `fa_radio_streamer` | 配信サンプル（Icecast等） | Sub: `audio/frame` → 外部へ送出（`ffmpeg`） |
| `fa_voice_command_router` | 起動/停止/モード切替の状態管理 | Sub: `voice/command` → Pub: `voice/router/state` / Srv: `start`, `stop`, `status` |

## 3. 将来のノード（予定）
- `src/ai/fa_kws`: ウェイクワード/キーワードスポッティング（オフライン）
- `src/ai/fa_asr`: 音声認識（オフライン）
- `src/ai/fa_sed`: 音イベント検出（オフライン）
- `src/ai/fa_speaker`: 話者認識（オフライン）
- `src/apps/fa_safety_policy`: 危険操作の拒否/確認要求など
- `src/dsp/*`, `src/features/*`: ノイズ抑制/AEC/特徴量など

## 4. インターフェース

### 4.1 トピック
| トピック | 型 | 送信元 | 用途 |
| --- | --- | --- | --- |
| `audio/frame` | `fa_interfaces/msg/AudioFrame` | `fa_capture_node` | PCM + メタ情報 |
| `audio/levels` | `std_msgs/msg/Float32MultiArray` | `fa_capture_node` | UIメータ表示 |
| `audio/vad` | `std_msgs/msg/Bool` | `fa_vad_node` | 簡易VADフラグ |
| `voice/vad_state` | `fa_interfaces/msg/VadState` | `fa_vad_node` | VAD確率/開始/終了（Silero） |
| `audio/output/frame` | `fa_interfaces/msg/AudioFrame` | `fa_tts` 等 | スピーカー再生用 |
| `audio/tts/frame` | `fa_interfaces/msg/AudioFrame` | `fa_tts` | TTS結果のPCM配信（任意） |

### 4.2 サービス
| サービス | 型 | サーバー | 内容 |
| --- | --- | --- | --- |
| `list_devices` | `fa_interfaces/srv/ListDevices` | `fa_capture_node` | マイク列挙 |
| `switch_device` | `fa_interfaces/srv/SwitchDevice` | `fa_capture_node` | マイク切替 |
| `record` | `fa_interfaces/srv/Record` | `fa_record` | 録音開始/停止 |
| `audio/output/play_file` | `fa_interfaces/srv/PlayFile` | `fa_output` | WAVファイル再生 |
| `speak` | `fa_interfaces/srv/Speak` | `fa_tts` | テキスト→音声 |

## 5. 運用フロー例

### 5.1 TTS をスピーカーへ再生
1. `fa_output`を起動
2. `fa_tts`を起動
3. `/speak` を呼び出し（`play: true`）、`audio/output/frame`経由で再生

### 5.2 マイク入力 + VAD
1. `fa_capture_node`を起動（`audio/frame`をPublish）
2. `fa_vad_node`を起動（`audio/vad`をPublish）

### 5.3 録音（WAV）
1. `fa_capture_node`と`fa_record`を起動
2. `record`サービスで開始/停止し、WAVを保存

## 6. 配信サンプル（Icecast向け）
`fa_stream`には`audio/frame`をIcecast/Shoutcastへ配信するPythonサンプル（`radio_streamer.py`）を同梱しています。内部で`ffmpeg`を起動し、受信したPCM16フレームをMP3等へ変換してHTTP PUTします。

起動例:
```bash
ros2 launch fa_stream fa_stream.launch.py
```
