# FluentVision オーディオシステム資料

`fv_audio`系ノードでマイク入力・スピーカー出力・ウェイクワード検知・TTS再生を統合するための資料。`fv_camera`/`fv_realsense`の設計思想を踏襲し、ROS2トピック/サービス/アクションを通じて組み合わせやすくする。

## 1. 目的
1. **入力/出力デバイスの分離**: マイクとスピーカーを別ノードにして、デバイス占有や遅延調整を独立管理する。
2. **音声イベント活用**: Wakeword検知や通知、アラートを音声ベースで扱い、映像系UIと同期させる。
3. **音声合成/TTS**: テキストから音声を出し、遠隔オペレータ通知やロボットフィードバックを音声化。
4. **再利用可能なインターフェース**: すべての音声は`fv_audio/msg/AudioFrame`や`audio/output/play`サービスに統一し、録画・配信・UIで共通利用できるようにする。

## 2. ノード構成

| ノード | 役割 | 主な入出力 |
| --- | --- | --- |
| `fv_audio_capture` | マイク/ライン入力を開きPCM配信 | Pub: `/audio/frame`, `/audio/levels` |
| `fv_audio_vad` | 軽量VAD（RMS/ヒステリシス） | Sub: `/audio/frame` → Pub: `/audio/vad` |
| `fv_wakeword_listener` | Wakeword/キーワード検知 | Sub: `/audio/frame`/`/audio/vad` → Pub: `/wake_event`, `/wake_detected` |
| `fv_audio_notifier` | 特定イベント時に効果音/TTSを再生 | Sub: `/wake_event` など → Call: `/audio/output/play` |
| `fv_tts` | テキスト→音声合成 | Service: `/fv_tts/speak` → Pub: `/audio/tts/frame` or Call `/audio/output/play` |
| `fv_audio_output` | スピーカーを開いてPCM/Opusを再生 | Sub: `/audio/output/frame` / Service: `/audio/output/play` |
| (オプション) `fv_audio_router` | 入出力ノードの仲介・ミキシング | 複数PCMをミックスして`fv_audio_output`へ送る |

## 3. ノード詳細

### 3.1 fv_audio_capture（入力ノード）
- ALSA APIでデバイス列挙・選択(`ListDevices`)、ホットスワップ(`SwitchDevice`)。
- QoS: `SensorDataQoS`。デフォルトは48kHz/モノラル/16bit/20ms chunk。
- RMS/Peakを算出しUIメータへPublish（VADは別ノード）。
- `Record`サービスでWAV保存。`fv_recorder`はこのサービスを呼び映像と同期記録。

### 3.2 fv_audio_vad
- `fv_audio/msg/AudioFrame`を購読し、RMSとヒステリシスで`std_msgs/Bool /audio/vad`を出力。
- パラメータ: `vad.threshold`, `vad.release_ms`, `vad.min_active_ms`。
- 将来的にWebRTC-VADやMLモデルへ差し替える場合もこのノードをリプレース。

### 3.3 fv_wakeword_listener
- 入力PCMをリングバッファに保持し、Porcupine/Snowboy/自前NN等を適用。
- パラメータ: `keyword_files`, `sensitivity`, `cooldown_ms`。
- 出力:
  - `std_msgs/Bool /wake_detected`（瞬時検知）
  - `fv_msgs/msg/WakeEvent`（ワード名、信頼度、タイムスタンプ、音声クリップパス）
- サービス: `SetSensitivity`, `AddKeyword` など。

### 3.4 fv_audio_notifier
- WakeEventをトリガに、以下の動作を選択：
  1. プリセット効果音再生（wav/ogg）
  2. `fv_tts`サービスでテキスト→音声化し再生
  3. `/audio/output/frame`へPCMを直接Publish
- 優先度キューを持ち、システムアラート→操作ガイダンスの順で再生。

### 3.5 fv_tts
- `fv_tts`はpyopenjtalk(Open JTalk)をバックエンドにした`rclpy`ノードとして実装済み。`ros2 run fv_tts fv_tts_node`または`ros2 launch fv_tts fv_tts.launch.py`で起動。
- `ros2 service call /fv_tts/speak "{text: 'こんにちは', voice_id: '', play: true}"`のように利用。`fv_tts/srv/Speak`のレスポンスには`fv_audio/msg/AudioFrame`が含まれ、そのまま録音や別ノードへ渡せる。
- `play: true`かつ`use_playback_topic`が`true`の場合は生成したPCMを`audio/output/frame`へPublishする。`false`の場合は`audio/tts/frame`だけにPublish。
- パラメータ: `default_voice`, `default_volume_db`, `cache_dir`, `output_topic`, `playback_topic`などを指定可能。
- キャッシュ: `~/.fluent_voice_cache`（デフォルト）へPCMとメタ情報を保存し、同じテキスト/ボイスは再利用。

### 3.6 fv_audio_output（出力ノード）
- PortAudioでスピーカーを開き、`/audio/output/frame`を購読、または`/audio/output/play`サービスでバイナリを受け取り再生。
- パラメータ: `device_selector`, `sample_rate`, `channels`, `latency_ms`, `mixing_enabled`。
- Mixモード: 複数PCM入力を同時再生。揮発的通知と長尺TTSを重ねる場合はミキサーを有効化。
- エラーハンドリング: XRUN検知、再接続、再生キュー詰まりのダイアグ報告。

### 3.7 fv_audio_router（任意）
- Captureの`audio/frame`を購読し、フィルタ/エンコーディング（Opus化など）を行ってWebSocket/RTMPに転送。
- 出力ノードの前段で音量正規化やオーディオダッキング（TTS発話中にマイク音量を自動調整）を行う場合にも利用。

## 4. インターフェースとデータフロー

### 4.1 トピック
| トピック | 型 | 送信元 | 用途 |
| --- | --- | --- | --- |
| `/fv/audio/<name>/audio/frame` | `fv_audio/msg/AudioFrame` | `fv_audio_capture` | PCM + メタ情報 |
| `/fv/audio/<name>/audio/levels` | `std_msgs/msg/Float32MultiArray` | `fv_audio_capture` | UIメータ表示 |
| `/fv/audio/<name>/audio/vad` | `std_msgs/msg/Bool` | `fv_audio_vad` | 簡易VADフラグ |
| `/fv/audio/<name>/wake_event` | `fv_msgs/msg/WakeEvent` | `fv_wakeword_listener` | 検知通知 |
| `/fv/audio/output/frame` | `fv_audio/msg/AudioFrame` | `fv_tts`, `fv_audio_notifier` 等 | スピーカー再生用 |
| `/fv/audio/tts/frame` | `fv_audio/msg/AudioFrame` | `fv_tts` | TTS結果のPCM配信 |

### 4.2 サービス
| サービス | 型 | サーバー | 内容 |
| --- | --- | --- | --- |
| `/fv_audio/list_devices` | `fv_audio/srv/ListDevices` | `fv_audio_capture` | マイク列挙 |
| `/fv_audio/switch_device` | `fv_audio/srv/SwitchDevice` | `fv_audio_capture` | マイク切替 |
| `/fv_audio/record` | `fv_audio/srv/Record` | `fv_audio_capture` | 録音開始/停止 |
| `/fv_audio_output/list_devices` | 同上 (別命名) | `fv_audio_output` | スピーカー列挙 |
| `/audio/output/play` | `fv_audio/srv/PlayAudio` (新規) | `fv_audio_output` | PCM再生リクエスト（ファイル/バッファ） |
| `/fv_tts/speak` | `fv_tts/srv/Speak` | `fv_tts` | テキスト→音声 |

### 4.3 アクション（必要に応じて）
- `fv_audio/action/CaptureClip`: Wakeword検知前後N秒をクリップ化し、解析ノードへ渡す。
- `fv_audio/action/PlaySequence`: 複数音声を順序再生（TTS→効果音など）。

## 5. 運用フロー例
1. **常時リスニング**: `fv_audio_capture`がマイクを開き、`fv_wakeword_listener`が常時解析。
2. **ウェイクワード検知**: 「フルーエント」と発話 → `WakeEvent` Publish。
3. **イベント通知**: `fv_audio_notifier`がWakeEventを受け、`fv_tts`へ「どうしましたか？」を依頼。
4. **TTS再生**: `fv_tts`がPCMを生成し`fv_audio_output`へ送信。スピーカーから応答。
5. **追加コマンド**: オペレータが音声コマンドを続けて発話 → 別ノードが`audio/frame`を解析。

## 6. 実装ロードマップ
1. **Phase 1**: `fv_audio_capture` + `fv_audio_output` の基本入出力整備。ALSA依存・サービス・診断を実装。
2. **Phase 2**: `fv_audio_vad`のPoC。閾値/ヒステリシスを実装し、Wakeword前段として利用。
3. **Phase 3**: `fv_wakeword_listener`のPoC。Porcupine等のライブラリラップ。`WakeEvent`メッセージ追加。
4. **Phase 4**: `fv_tts`（Python/C++）でTTSサービス化。VOICEVOX/Coqui/Pollyなど実装差し替え可能な構造にする。
5. **Phase 5**: `fv_audio_notifier`でWakeword/システムイベント→音声通知を構築。CLI (`./fv audio start|stop|status`) と連携。
6. **Phase 6**: Webダッシュボード/RTMPへの音声統合、録画との同期検証、長時間運用テスト。

## 7. テスト/検証指針
- **ユニット**: PCM→RMS/VAD計算、ALSAデバイス選択ロジック、TTSキャッシュ。
- **結合**: 仮想ループバック (`snd-aloop`) で`capture`→`wakeword`→`output`を自動テスト。
- **システム**: 連続24時間運用でXRUN/遅延監視。Wakeword誤検知率とTTS遅延をログ化。

## 8. インターネットラジオ配信サンプル

`fv_audio`には`fv_audio/msg/AudioFrame`をIcecast/Shoutcastへ配信するPythonサンプル（`scripts/radio_streamer.py`）を同梱した。内部で`ffmpeg`を起動し、受信したPCM16フレームをMP3へ変換してHTTP PUTする。

1. Icecastサーバを用意（例: `sudo apt install icecast2`、`/etc/icecast2/icecast.xml`で `source` ユーザとマウント `/live` を許可）。
2. `fv_audio_node`を起動し、`audio/frame`トピックへPCMをPublishする。
3. ストリーマーを実行:
   ```bash
   ros2 run fv_audio radio_streamer.py \
     --ros-args \
     -p output_url:=http://source:hackme@localhost:8000/live \
     -p bitrate:=128k \
     -p container_format:=mp3 \
     -p audio_codec:=libmp3lame
   ```
4. クライアント側は `http://<host>:8000/live` を再生。`bitrate`, `audio_codec`, `content_type` 等はROSパラメータで調整できる。

備考:
- 現状16ビットPCMのみ対応（`fv_audio`の`bit_depth`を16に設定）。
- `ffmpeg`バイナリがインストールされている必要がある（`sudo apt install ffmpeg`）。
- Icecast以外のHTTPエンドポイントへ送る場合は`output_url`を書き換えるだけで応用可能。

この資料をベースに、`fv_audio`関連パッケージを段階的に実装すれば、入力・解析・通知・合成・出力まで一貫したオーディオパイプラインを構築できる。
