# FluentAudio ROS2

`fluent_vision_ros2`（設計・開発: Takashi Otsuka / @takatronix）をベースに、ROS2 上で「聴覚（Audio）」を扱うためのノード群を整理した repository です。

- Upstream: https://github.com/takatronix/fluent_vision_ros2
- This fork: https://github.com/Escenda/fluent_audio_ros2

## このリポジトリの位置づけ
- upstream の設計思想（ノード分割、低遅延、YAML/launch 運用）を踏襲します
- `src/` 以下を用途別（`io/`, `processing/`, `ai/`, `streaming/`, `apps/`, `system/`, `interfaces/`）に分割して管理します
- source / sink、DSP、音声 AI、音声 app を分離し、backend を明示的に切り替えられる構造を目標にしています
- 音声にフォーカスするため、vision 系パッケージ（カメラ/AI/UI/配信/SLAM など）は本リポジトリから削除しています（視覚系は upstream を参照してください）

## 主要パッケージ（Audio）
- `fa_in`（`src/io/sources/fa_in/`）: 明示 backend で選択した入力源（ALSA / raw PCM file / raw PCM UDP）→ `audio/frame` を Publish、Diagnostics
- `fa_out`（`src/io/sinks/fa_out/`）: 明示 backend で選択した出力先（ALSA / raw PCM file / raw PCM UDP）へ `AudioFrame` を出力
- `fa_record`（`src/io/utilities/fa_record/`）: `audio/frame` をWAVへ録音（`record` サービス）
- `fa_stream`（`src/io/utilities/fa_stream/`）: `audio/frame` を外部へ配信する utility（Icecast向け `fa_stream_node.py`）
- `fa_vad`（`src/ai/fa_vad/`）: Silero VAD（PyTorch）で`audio/vad`と source / stream identity 付き `voice/vad_state` を提供
- `fa_kws`（`src/ai/fa_kws/`）: 外部 worker 境界の sherpa-onnx KWS、`voice/wake_word`を提供
- `fa_asr`（`src/ai/fa_asr/`）: ローカルASRコマンド（whisper.cpp等）を呼び出し、`voice/asr/result`を提供
- `fa_turn_detector`（`src/ai/fa_turn_detector/`）: Smart Turn v3 ONNX によるターン終了推定、`voice/turn_end`を提供
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
- VAD/ASR/TD: `python3-numpy`, 外部 VAD worker command, `onnxruntime`（TD）, ローカルASR実行ファイル（例: whisper.cpp）
- KWS: 外部 sherpa-onnx worker command（例: `fa_kws/scripts/sherpa_onnx_kws_worker`）
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
ros2 launch fa_out fa_out.launch.py node_name:=fa_out config_file:=/path/to/fa_out.yaml

# Terminal B
ros2 launch fa_tts fa_tts.launch.py node_name:=fa_tts config_file:=/path/to/fa_tts.yaml

# Terminal C
ros2 launch fa_mix fa_mix.launch.py node_name:=fa_mix config_file:=/path/to/fa_mix.yaml

# Terminal D（サービス名は namespace により変わる場合があります。`ros2 service list | grep speak` で確認）
ros2 service call /speak fa_interfaces/srv/Speak "{text: 'こんにちは', voice_id: '', play: false, volume_db: 0.0, cache_key: ''}"
```
`fa_tts` は `audio/tts/frame` だけを出力します。スピーカー再生は `fa_mix` で `audio/output/frame` へ routing してから `fa_out` が行います。

### 2) マイク入力 + VAD
```bash
ros2 launch fa_in fa_in.launch.py node_name:=fa_in config_file:=/path/to/fa_in.yaml
ros2 launch fa_vad fa_vad.launch.py node_name:=fa_vad config_file:=/path/to/fa_vad.yaml
```

### 3) SO101 VAD/KWS frontend
```bash
ros2 launch fluent_audio_system run.py \
  config:=/path/to/so101_kws_frontend.yaml \
  fa_in_enabled:=true \
  fa_out_enabled:=false \
  fa_in_source_id:=hw:CARD=Mic,DEV=0 \
  fa_out_sink_id:=disabled
```

この profile は `fa_in -> fa_sample_format -> fa_resample -> fa_vad / fa_kws` を起動します。ASR / Turn Detector は package-owned SO101 profile template に backend contract を持ちますが、この KWS frontend profile では起動しません。
ASR / Turn Detector まで同時起動する検証用 profile は `so101_voice_frontend.yaml` です。この profile も会話 orchestration は含まず、`conversation/turn_context` は上位 app から publish される前提です。

`fa_vad` / `fa_kws` / `fa_asr` / `fa_turn_detector` はローカルモデルまたは external worker の契約が必須です。VAD は `backend.command`、`backend.args`、model path、provider、workspace、QoS を明示します。KWS / ASR / Turn Detector はそれに加えて `backend.health_args` も node config または system config に明示します。未設定または存在しない場合は起動時に失敗します。

`voice/vad_state` は `fa_vad` が判定した `AudioFrame.source_id` / `stream_id` を保持します。`fa_kws` / `fa_asr` / `fa_turn_detector` は topic 名だけで VAD state を信頼せず、自分が処理する audio stream と identity が一致しない VAD state を reject します。

## インターフェース（抜粋）
- Topics:
  - `audio/frame`（`fa_interfaces/msg/AudioFrame`）
  - `audio/vad`（`std_msgs/msg/Bool`）
  - `voice/vad_state`（`fa_interfaces/msg/VadState`: VAD probability / start / end / source_id / stream_id）
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
