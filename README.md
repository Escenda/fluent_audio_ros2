# FluentAudio ROS2

`fluent_vision_ros2`（設計・開発: Takashi Otsuka / @takatronix）をベースに、ROS2 上で「聴覚（Audio）」を扱うためのノード群を整理した repository です。

- Upstream: https://github.com/takatronix/fluent_vision_ros2
- This fork: https://github.com/Escenda/fluent_audio_ros2

## このリポジトリの位置づけ
- upstream の設計思想（ノード分割、低遅延、YAML/launch 運用）を踏襲します
- `src/` 以下を用途別（`io/`, `processing/`, `apps/`, `system/`, `interfaces/`）に分割して管理します
- source / sink、DSP、音声 AI、音声 app を分離し、backend を明示的に切り替えられる構造を目標にしています
- 音声にフォーカスするため、vision 系パッケージ（カメラ/AI/UI/配信/SLAM など）は本リポジトリから削除しています（視覚系は upstream を参照してください）

## 主要パッケージ（Audio）
- `fa_in`（`src/io/sources/fa_in/`）: マイク入力（ALSA）→ `audio/frame` を Publish、デバイス列挙/切替、Diagnostics
- `fa_out`（`src/io/sinks/fa_out/`）: `audio/output/frame` をスピーカーへ再生
- `fa_record`（`src/io/utilities/fa_record/`）: `audio/frame` をWAVへ録音（`record` サービス）
- `fa_stream`（`src/io/utilities/fa_stream/`）: `audio/frame` を外部へ配信する utility（Icecast向け `fa_stream_node.py`）
- `fa_vad`（`src/processing/analysis/fa_vad/`）: Silero VAD（PyTorch）で`audio/vad`と`voice/vad_state`を提供
- `fa_kws`（`src/processing/analysis/fa_kws/`）: sherpa-onnx によるローカルKWS、`voice/wake_word`を提供
- `fa_asr`（`src/processing/analysis/fa_asr/`）: ローカルASRコマンド（whisper.cpp等）を呼び出し、`voice/asr/result`を提供
- `fa_turn_detector`（`src/processing/analysis/fa_turn_detector/`）: Smart Turn v3 ONNX によるターン終了推定、`voice/turn_end`を提供
- `fa_tts`（`src/processing/generation/fa_tts/`）: pyopenjtalk(Open JTalk) によるTTS（`speak` サービス）/ `AudioFrame` 出力
- `fa_resample`（`src/processing/format/fa_resample/`）: 16k ストリーム供給（`audio/frame`→`audio/resample16k/mic`）
- `fa_aec_linear`（`src/processing/correction/fa_aec_linear/`）: 線形AEC（`mic/ref`→`audio/aec_linear/frame`）
- `fa_aec_nn`（`src/processing/correction/fa_aec_nn/`）: NN残差抑圧（骨組み、`audio/aec_linear/frame`→`audio/aec/frame`）
- `fa_denoise`（`src/processing/correction/fa_denoise/`）: ノイズ抑制（DTLN/ONNX、`audio/resample16k/mic`→`audio/denoise/frame`）
- `fa_mix`（`src/processing/routing/fa_mix/`）: ミキサ（MVP、`input_topics`→`audio/output/frame`）
- `fa_voice_command_router`（`src/apps/voice_command/fa_voice_command_router/`）: 音声コマンドの起動/停止/モード切替（MVP: 文字列コマンド入力）
- `fa_interfaces`（`src/interfaces/fa_interfaces/`）: `AudioFrame` 等の msg/srv を集約

## セットアップ

### 前提
- ROS 2（Humble/Jazzy など）
- ALSA: `libasound2-dev`
- TTS: `pyopenjtalk`, `python3-numpy`
- VAD/ASR/TD: `python3-numpy`, `torch`（VAD）, `onnxruntime`（TD）, ローカルASR実行ファイル（例: whisper.cpp）
- KWS: sherpa-onnx C API
- （任意）`ffmpeg`: `fa_stream` の `fa_stream_node.py` で使用

### ビルド
```bash
colcon build --symlink-install
source install/setup.bash
```

## クイックスタート

### 1) TTS をスピーカーへ再生
```bash
# Terminal A
ros2 launch fa_out fa_out.launch.py

# Terminal B
ros2 launch fa_tts fa_tts.launch.py

# Terminal C
ros2 launch fa_mix fa_mix.launch.py

# Terminal D（サービス名は namespace により変わる場合があります。`ros2 service list | grep speak` で確認）
ros2 service call /speak fa_interfaces/srv/Speak "{text: 'こんにちは', voice_id: '', play: false, volume_db: 0.0, cache_key: ''}"
```
`fa_tts` は `audio/tts/frame` だけを出力します。スピーカー再生は `fa_mix` で `audio/output/frame` へ routing してから `fa_out` が行います。

### 2) マイク入力 + VAD
```bash
ros2 launch fa_in fa_in.launch.py
ros2 launch fa_vad fa_vad.launch.py
```

### 3) 音声対話コア（VAD/KWS/ASR/TD）
```bash
ros2 launch fa_in fa_in.launch.py
ros2 launch fa_vad fa_vad.launch.py
ros2 launch fa_kws fa_kws.launch.py
ros2 launch fa_asr fa_asr.launch.py
ros2 launch fa_turn_detector fa_turn_detector.launch.py
```

`fa_kws` / `fa_asr` / `fa_turn_detector` はローカルモデルのパスが必須です。未設定または存在しない場合は起動時に失敗します。

## インターフェース（抜粋）
- Topics:
  - `audio/frame`（`fa_interfaces/msg/AudioFrame`）
  - `audio/vad`（`std_msgs/msg/Bool`）
  - `voice/vad_state`（`fa_interfaces/msg/VadState`）
  - `voice/wake_word`（`fa_interfaces/msg/WakeWordResult`）
  - `voice/asr/result`（`fa_interfaces/msg/AsrResult`）
  - `voice/turn_end`（`fa_interfaces/msg/TurnEnd`）
  - `conversation/turn_context`（`fa_interfaces/msg/TurnContext`）
  - `audio/output/frame`（`fa_interfaces/msg/AudioFrame`）
- Services:
  - `list_devices`, `switch_device`（`fa_in`）
  - `record`（`fa_record`）
  - `speak`（`fa_tts`）

## ドキュメント
- `docs/仕様書.md`: repository 全体の責務境界
- `docs/アルゴリズム詳細説明書.md`: package 横断の処理分類と backend 境界
- `docs/テスト設計.md`: package 横断のテスト方針
- `docs/fa_audio_system.md`: 全体像・データフロー
- `docs/fa_audio_design.md`: 設計メモ
- 各 package 配下の `README.md` と `docs/`

## ライセンス
このリポジトリには MIT / Apache-2.0 など複数ライセンスのパッケージが含まれます。各パッケージの `package.xml` の `<license>` を参照してください。

## クレジット
- Original project: Takashi Otsuka (@takatronix) / FluentVision ROS2
- This fork: Escenda + contributors
