# FluentAudio ROS2

`fluent_vision_ros2`（設計・開発: Takashi Otsuka / @takatronix）をベースに、ROS2上で「聴覚（Audio）」を扱うためのノード群を追加したフォークです。

- Upstream: https://github.com/takatronix/fluent_vision_ros2
- This fork: https://github.com/Escenda/fluent_audio_ros2

## このリポジトリの位置づけ
- upstream の設計思想（ノード分割、低遅延、YAML/launch 運用）を踏襲します
- `src/` 以下を用途別（`io/`, `dsp/`, `features/`, `ai/`, `apps/`, `interfaces/`）に分割して管理します
- クラウドを使わず、音声入力で「起動/停止/モード切替」できる運用を目標にしています
- 音声にフォーカスするため、vision 系パッケージ（カメラ/AI/UI/配信/SLAM など）は本リポジトリから削除しています（視覚系は upstream を参照してください）

## 主要パッケージ（Audio）
- `fa_capture`（`src/io/fa_capture/`）: マイク入力（ALSA）→ `audio/frame` を Publish、デバイス列挙/切替、Diagnostics
- `fa_output`（`src/io/fa_output/`）: `audio/output/frame` をスピーカーへ再生、`audio/output/play_file` でWAVを直接再生
- `fa_record`（`src/io/fa_record/`）: `audio/frame` をWAVへ録音（`record` サービス）
- `fa_stream`（`src/io/fa_stream/`）: `audio/frame` を外部へ配信するサンプル（Icecast向け `radio_streamer.py`）
- `fa_vad`（`src/ai/fa_vad/`）: Silero VAD（PyTorch）で`audio/vad`と`voice/vad_state`を提供
- `fa_tts`（`src/ai/fa_tts/`）: pyopenjtalk(Open JTalk) によるTTS（`speak` サービス）/ `AudioFrame` 出力
- `fa_voice_command_router`（`src/apps/fa_voice_command_router/`）: 音声コマンドの起動/停止/モード切替（MVP: 文字列コマンド入力）
- `fa_interfaces`（`src/interfaces/fa_interfaces/`）: `AudioFrame` 等の msg/srv を集約

## セットアップ

### 前提
- ROS 2（Humble/Jazzy など）
- ALSA: `libasound2-dev`
- TTS: `pyopenjtalk`, `python3-numpy`
- （任意）`ffmpeg`: `fa_stream` の `radio_streamer.py` サンプルで使用

### ビルド
```bash
colcon build --symlink-install
source install/setup.bash
```

## クイックスタート

### 1) TTS をスピーカーへ再生
```bash
# Terminal A
ros2 launch fa_output fa_output.launch.py

# Terminal B
ros2 launch fa_tts fa_tts.launch.py

# Terminal C（サービス名は namespace により変わる場合があります。`ros2 service list | grep speak` で確認）
ros2 service call /speak fa_interfaces/srv/Speak "{text: 'こんにちは', voice_id: '', play: true}"
```

### 2) マイク入力 + VAD
```bash
ros2 launch fa_capture fa_capture.launch.py
ros2 launch fa_vad fa_vad.launch.py
```

## インターフェース（抜粋）
- Topics:
  - `audio/frame`（`fa_interfaces/msg/AudioFrame`）
  - `audio/levels`（`std_msgs/msg/Float32MultiArray`）
  - `audio/vad`（`std_msgs/msg/Bool`）
  - `voice/vad_state`（`fa_interfaces/msg/VadState`）
  - `audio/output/frame`（`fa_interfaces/msg/AudioFrame`）
- Services:
  - `list_devices`, `switch_device`（`fa_capture`）
  - `record`（`fa_record`）
  - `audio/output/play_file`（`fa_output`）
  - `speak`（`fa_tts`）

## ドキュメント
- `docs/fa_audio_system.md`: 全体像・データフロー
- `docs/fa_audio_design.md`: 設計メモ
- 各パッケージ配下の `README.md`

## ライセンス
このリポジトリには MIT / Apache-2.0 など複数ライセンスのパッケージが含まれます。各パッケージの `package.xml` の `<license>` を参照してください。

## クレジット
- Original project: Takashi Otsuka (@takatronix) / FluentVision ROS2
- This fork: Escenda + contributors
