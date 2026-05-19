# FluentAudio オーディオシステム資料

本資料は FluentAudio（本リポジトリ）の音声パッケージ群を、ROS2のトピック/サービス観点で俯瞰するためのメモです。

## 1. 目的（制約込み）
1. **入力/出力デバイスの分離**: マイクとスピーカーを別ノードにして、デバイス占有や遅延調整を独立管理する。
2. **低遅延・組み合わせ容易性**: upstream と同様に、疎結合（topic/srv）で差し替え可能な構成にする。
3. **オフライン前提**: クラウドを使わず、音声入力で「起動/停止/モード切替」を実現できる土台を作る。
4. **再利用可能なインターフェース**: 音声フレームは `fa_interfaces/msg/AudioFrame` に統一する。

## 2. ノード構成（現在のROS 2 package）

この表は package / topic contract の俯瞰であり、全 package の DSP / backend 実装完了を意味しない。
passthrough contract や skeleton 実装の package は、各 package の `docs/仕様書.md` と `docs/backends/` を正とする。

| ノード | 役割 | 主な入出力 |
| --- | --- | --- |
| `fa_in` | 明示 source からPCMを配信 | Pub: `audio/frame` など / Srv: `list_devices`, `switch_device` |
| `fa_vad` | Silero VAD | Sub: `audio/frame` → Pub: `audio/vad`, `voice/vad_state` |
| `fa_kws` | ローカルKWS | Sub: `audio/frame`, `voice/vad_state` → Pub: `voice/wake_word` |
| `fa_asr` | ローカルASR実行ファイルの呼び出し | Sub: `audio/frame`, `voice/vad_state`, `conversation/turn_context` → Pub: `voice/asr/result` |
| `fa_turn_detector` | Smart Turn v3 ONNX によるターン終了推定 | Sub: `audio/frame`, `voice/vad_state`, `conversation/turn_context` → Pub: `voice/turn_end` |
| `fa_record` | `audio/frame` をWAVへ録音 | Sub: `audio/frame` / Srv: `record` |
| `fa_tts` | テキスト→音声合成 | Srv: `speak` → Pub: `audio/tts/frame` |
| `fa_mix` | routing / mixing | Sub: `audio/tts/frame` 等 → Pub: `audio/output/frame` |
| `fa_out` | スピーカーを開いてPCMを再生 | Sub: `audio/output/frame` |
| `fa_stream` | 配信 sink（Icecast等） | Sub: `audio/frame` → 外部へ送出（`ffmpeg`） |
| `fa_voice_command_router` | 起動/停止/モード切替の状態管理 | Sub: `voice/command` → Pub: `voice/router/state` / Srv: `start`, `stop`, `status` |

## 3. 将来のノード（予定）
- `src/ai/fa_sed`: 音イベント検出（オフライン）
- `src/ai/fa_speaker`: 話者認識（オフライン）
- `src/apps/dialogue/fa_dialogue`: Wakeword/ASR/TD を合流する会話オーケストレーション
- `src/apps/safety/fa_safety_policy`: 危険操作の拒否/確認要求など
- `src/processing/<category>/*`: ノイズ抑制/AEC/特徴量などの DSP / feature extraction
- `src/streaming/*`: jitter buffer、clock drift、PLC、time alignment などのリアルタイム伝送安定化

## 4. インターフェース

### 4.1 トピック
| トピック | 型 | 送信元 | 用途 |
| --- | --- | --- | --- |
| `audio/frame` | `fa_interfaces/msg/AudioFrame` | `fa_in` | PCM + メタ情報 |
| `audio/vad` | `std_msgs/msg/Bool` | `fa_vad` | 簡易VADフラグ |
| `voice/vad_state` | `fa_interfaces/msg/VadState` | `fa_vad` | VAD確率/開始/終了と判定元 source / stream identity |
| `voice/wake_word` | `fa_interfaces/msg/WakeWordResult` | `fa_kws` | ウェイクワード検出 |
| `conversation/turn_context` | `fa_interfaces/msg/TurnContext` | 会話オーケストレータ | ASR/TDのID相関 |
| `voice/asr/result` | `fa_interfaces/msg/AsrResult` | `fa_asr` | ASR結果/タイムアウト/エラー |
| `voice/turn_end` | `fa_interfaces/msg/TurnEnd` | `fa_turn_detector` | ターン終了確率 |
| `audio/output/frame` | `fa_interfaces/msg/AudioFrame` | `fa_mix` 等 | スピーカー再生用 |
| `audio/tts/frame` | `fa_interfaces/msg/AudioFrame` | `fa_tts` | TTS結果のPCM配信 |

### 4.2 サービス
| サービス | 型 | サーバー | 内容 |
| --- | --- | --- | --- |
| `list_devices` | `fa_interfaces/srv/ListDevices` | `fa_in` | マイク列挙 |
| `switch_device` | `fa_interfaces/srv/SwitchDevice` | `fa_in` | マイク切替 |
| `record` | `fa_interfaces/srv/Record` | `fa_record` | 録音開始/停止 |
| `speak` | `fa_interfaces/srv/Speak` | `fa_tts` | テキスト→音声 |

## 5. 運用フロー例

### 5.1 TTS をスピーカーへ再生
1. `fa_out`を起動
2. `fa_tts`を起動
3. `fa_mix`を起動し、`audio/tts/frame`を`audio/output/frame`へ routing
4. `/speak` を呼び出し（`play: false`）、`audio/tts/frame -> fa_mix -> audio/output/frame`経由で再生

### 5.2 マイク入力 + VAD
1. `fa_in`を起動（`audio/frame`をPublish）
2. `fa_vad`を起動（`audio/vad`をPublish）

### 5.3 VAD/KWS/ASR/TD
1. `fa_in`と`fa_vad`を起動
2. `fa_kws`を起動し、`voice/wake_word`で起動語を受ける
3. 会話オーケストレータが`conversation/turn_context`をPublishする
4. `fa_asr`が`voice/asr/result`をPublishする
5. `fa_turn_detector`が`voice/turn_end`をPublishする

この経路では `fa_vad` の入力 stream と、`fa_kws` / `fa_asr` / `fa_turn_detector` が処理する audio stream を一致させる必要があります。`VadState.source_id` / `stream_id` が一致しない場合、後段 node はその VAD state を gate / finalize / turn-end trigger として使いません。

SO101 の VAD + KWS frontend は `fluent_audio_system/config/profiles/so101_kws_frontend.yaml` に system config として定義します。VLAbor profile には enable / config path / source binding だけを置き、Silero / sherpa-onnx の worker command、model path、provider、keywords file は system config 側の `${env:...}` で明示します。

### 5.4 録音（WAV）
1. `fa_in`と`fa_record`を起動
2. `record`サービスで開始/停止し、WAVを保存

## 6. 配信 sink（Icecast向け）
`fa_stream`には`audio/frame`をIcecast/Shoutcastへ配信するPython node（`fa_stream_node.py`）を同梱しています。ROS2 node は topic / parameter / `AudioFrame` validation を担当し、`ffmpeg` の起動と書き込みは ROS-free な network streamer backend が担当します。`output_url` は必須で、空のままでは起動失敗します。

`fa_stream` は network sink utility であり、`src/streaming` の jitter buffer / clock drift / PLC などのリアルタイム伝送安定化 node ではありません。

起動例:
```bash
ros2 launch fa_stream fa_stream.launch.py
```
