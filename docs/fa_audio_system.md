# FluentAudio オーディオシステム資料

本資料は FluentAudio（本リポジトリ）の音声パッケージ群を、ROS2のトピック/サービス観点で俯瞰するためのメモです。

## 1. 目的（制約込み）
1. **入力/出力デバイスの分離**: マイクとスピーカーを別ノードにして、デバイス占有や遅延調整を独立管理する。
2. **低遅延・組み合わせ容易性**: upstream と同様に、疎結合（topic/srv）で差し替え可能な構成にする。
3. **明示 backend 前提**: local model、external worker、cloud API を同じ backend 境界で扱い、`backend.name` と必要な model / endpoint / credential を config で明示する。
4. **再利用可能なインターフェース**: 音声フレームは `fa_interfaces/msg/AudioFrame` に統一する。

## 2. 完了済み runtime package

この節には、機能実装、fail-closed validation、launch 経路、代表テスト、必要な実 backend / worker / model provisioning が揃い、現在の検証記録で完了扱いできる runtime package だけを記載する。
`package.xml`、README、launch file、topic contract、docs、skeleton、passthrough backend の存在だけでは完了済みに数えない。

| package | 完了範囲 | 代表検証 |
| --- | --- | --- |
| 現時点では未記載 | 完了判定には最新の代表検証が必要 | 未検証 |

## 3. 実装中 / 代表検証待ち

以下は runtime package または node 実装が存在するが、この資料では完了済みとして扱わない。完了済み一覧へ移すには、各 package の実装範囲に応じた代表検証を実行し、検証結果を残す必要がある。

| package | 現在確認できる実装範囲 | 完了扱いしない理由 |
| --- | --- | --- |
| `fa_in` | `alsa_capture`、`pcm_file_reader`、`network_pcm_receiver` source backend | 実 device / file / network source を含む代表 graph 起動と VLAbor site binding 経路が未検証 |
| `fa_out` | `alsa_playback`、`pcm_file_writer`、`network_pcm_sender` sink backend | 実 speaker / file / network sink と playback control を含む代表 graph 起動が未検証 |
| `fa_vad` | Silero external worker 境界と VAD state publish | 実 model / worker / provider provisioning を含む SO101 graph が未検証 |
| `fa_kws` | sherpa-onnx external worker 境界と wake word publish | 実 sherpa-onnx model / keywords / worker provisioning を含む SO101 graph が未検証 |
| `fa_asr` | `local_command`、`whisper.cpp`、`parakeet_worker`、`openai_realtime`、`openai_transcriptions` backend 枠 | 実 worker command / model / credential / health check を含む代表 graph が未検証 |
| `fa_turn_detector` | Smart Turn ONNX external worker 境界と turn end publish | 実 model / worker / provider provisioning と `fa_dialogue` 接続済み graph が未検証 |
| `fa_dialogue` | `WakeWordResult` / `AsrResult` / `TurnEnd` から `TurnContext` を publish する最小 turn context publisher。Docker/ROS smoke で WakeWord -> active `TurnContext`、ASR terminal / TurnEnd terminal -> inactive `TurnContext` の pub/sub 契約、および `ros2 launch fa_dialogue fa_dialogue.launch.py config_file:=<tmp_yaml>` による config file routing を確認済み | reasoning / TTS / safety policy / external dialogue backend / robot command proposal / full SO101 model provisioning / 親 VLAbor profile integration は未実装または未検証 |
| `fa_record` | `AudioFrame` の WAV 録音 utility | `fa_in` からの実 graph 録音 smoke が未検証 |
| `fa_stream` | `ffmpeg` network streamer utility | 実 endpoint への streaming smoke と transport failure contract が未検証 |
| `fa_tts` | TTS service と `audio/tts/frame` publish | 実 TTS backend、`fa_mix`、`fa_out` 連携が未検証 |
| `fa_mix` | PCM16LE の MVP mixing | routing / ducking / limiter / barge-in 連携は未完了 |
| `fa_voice_command_router` | MVP command router | structured command schema と KWS/ASR/TD 連携が未完了 |
| `fa_audio_mcp` | `archive_audio_window` / `transcribe_audio` MCP adapter、SO101 agent audio tools profile、in-process `FastMCP` から実 `fa_asr_node` / `fa_audio_window_node` を呼ぶ Docker runtime smoke、installed server に streamable-http MCP transport で接続して実 owner node へ tool call する smoke、`so101_voice_frontend.yaml,so101_agent_audio_tools.yaml` の child-side FluentAudio config composition / required package order 検証、`owner.yaml,adapter.yaml` multi-config list を `ros2 launch fluent_audio_system run.py` に渡して `fa_asr` / `fa_audio_window` owner services と `fa_audio_mcp_server` adapter executable を同時起動する Docker ROS runtime smoke | 実 Agent Runtime / MCP client、親 VLAbor profile integration、durable storage / World Station 連携、実 SO101 device / model provisioning の検証完了は意味しない |

## 4. 設計枠 / package 化前

以下は package 化前または roadmap placeholder として扱う。`fluent_audio_system` config から参照してはならない。

| path | 位置づけ |
| --- | --- |
| `src/ai/fa_sed` | 音イベント検出。package 化前 |
| `src/ai/fa_speaker` | 話者認識。package 化前 |
| `src/apps/safety/fa_safety_policy` | 危険操作の拒否 / 確認要求などの safety policy。package 化前 |
| `src/io/sources/fa_file_in`, `src/io/sources/fa_network_in` | 現在は `fa_in` backend として実装。standalone package 化前 |
| `src/io/sinks/fa_file_out`, `src/io/sinks/fa_network_out` | 現在は `fa_out` backend として実装。standalone package 化前 |
| `src/processing/*` の README-only directory | `docs/roadmap_placeholders.md` の分類に従う |

## 4. インターフェース

### 4.1 トピック
| トピック | 型 | 送信元 | 用途 |
| --- | --- | --- | --- |
| `audio/frame` | `fa_interfaces/msg/AudioFrame` | `fa_in` | PCM + メタ情報 |
| `audio/vad` | `std_msgs/msg/Bool` | `fa_vad` | 簡易VADフラグ |
| `voice/vad_state` | `fa_interfaces/msg/VadState` | `fa_vad` | VAD確率/開始/終了と判定元 source / stream identity |
| `voice/wake_word` | `fa_interfaces/msg/WakeWordResult` | `fa_kws` | ウェイクワード検出 |
| `conversation/turn_context` | `fa_interfaces/msg/TurnContext` | `fa_dialogue` | ASR/TDのID相関 |
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

### 5.3 SO101 VAD/KWS frontend
1. `fa_in` が raw microphone frame を publish する
2. `fa_sample_format` が `FLOAT32LE/32/interleaved` へ明示変換する
3. `fa_resample` が 16kHz stream へ明示変換する
4. `fa_vad` が `voice/vad_state` を publish する
5. `fa_kws` が `voice/wake_word` を publish する

SO101 の VAD + KWS frontend は `fluent_audio_system/config/profiles/so101_kws_frontend.yaml` に system config として定義します。この profile は `fa_asr` / `fa_turn_detector` を起動しません。VLAbor profile には enable / config path / source binding だけを置き、Silero / sherpa-onnx の worker command、model path、provider、keywords file は system config 側の `${env:...}` で明示します。

VAD / KWS / ASR / Turn Detector は `FLOAT32LE/32/interleaved`、configured sample rate、mono などの supported AudioFrame contract を各 node/backend で検証します。`PCM16LE` や sample-rate mismatch を受けた場合に AI node 内で resample / format conversion / downmix は行いません。必要な変換は `fa_sample_format`、`fa_resample`、`fa_channel_convert` を pipeline に明示し、未対応入力は frame rejection または fail-closed として扱います。

### 5.4 VAD/KWS/ASR/TD dialogue graph
1. VAD/KWS frontend を起動し、`voice/wake_word` で起動語を受ける
2. `fa_dialogue` が `conversation/turn_context` を publish する
3. `fa_asr` が `voice/asr/result` を publish する
4. `fa_turn_detector` が `voice/turn_end` を publish する

この経路では `fa_vad` の入力 stream と、`fa_kws` / `fa_asr` / `fa_turn_detector` が処理する audio stream を一致させる必要があります。`VadState.source_id` / `stream_id` が一致しない場合、後段 node はその VAD state を gate / finalize / turn-end trigger として使いません。ASR / Turn Detector の backend command、model path、provider、endpoint、credential env、health args は、それらを enabled にする FluentAudio system config 側に閉じます。
SO101 で VAD/KWS/ASR/TD と `fa_dialogue` をまとめて起動する package-owned profile は `fluent_audio_system/config/profiles/so101_voice_frontend.yaml` です。この profile は wake word 後の session / turn 制御として `fa_dialogue` を起動しますが、LLM reasoning、TTS、safety policy、robot command proposal は含みません。

`src/apps/dialogue/fa_dialogue/test/integration/test_fa_dialogue_ros_graph_contract.py` は、Docker/ROS 環境で `fa_dialogue` rebuild 後に `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/apps/dialogue/fa_dialogue/test/integration/test_fa_dialogue_ros_graph_contract.py -q` を実行し、`3 passed in 1.65s` として確認済みです。`test_fa_dialogue_ros_graph_publishes_active_then_inactive_turn_context` は real `fa_dialogue_node` を `ros2 run` で起動し、`WakeWordResult` publish 後の active `TurnContext`、`AsrResult.STATUS_FINAL` と `TurnEnd.is_end=true` の両 terminal 経路による inactive `TurnContext` を pub/sub として検証します。`test_fa_dialogue_launch_uses_config_file_and_publishes_turn_context` は `ros2 launch fa_dialogue fa_dialogue.launch.py config_file:=<tmp_yaml>` で起動し、launch 経由で渡した config file の topic 設定に従って active `TurnContext` と ASR FINAL 後の inactive `TurnContext` を観測します。local non-ROS shell では `PYTHONPATH=src/apps/dialogue/fa_dialogue pytest -q src/apps/dialogue/fa_dialogue/test` が `13 passed, 1 skipped` です。これは `fa_dialogue` turn-context publisher と launch config file routing の smoke であり、reasoning、TTS、safety policy、external dialogue backend、robot command proposal、full SO101 model provisioning、親 VLAbor profile integration の検証ではありません。

### 5.5 Agent audio tools
`fluent_audio_system/config/profiles/so101_agent_audio_tools.yaml` は `fa_audio_mcp` を起動し、`archive_audio_window` と `transcribe_audio` を Agent / MCP client から呼べる入口を用意します。この profile は `fa_asr` や `fa_audio_window` の service owner を起動しません。通常は `so101_voice_frontend.yaml` と同じ親側 include 方式で組み合わせます。

`fluent_audio_system` の `config` / `--config` は、単一 system YAML に加えて、カンマ区切りの明示 config path list を受け付けます。複数指定時は左から右の順に child-side FluentAudio config を compose し、group / node / required package の順序を保持します。composition は fail closed で、空 segment、missing config、invalid YAML、`system.default_start_delay` / `system.inter_group_delay` 不一致、group id 重複、enabled node id 重複を成功扱いしません。

この config composition は child repo の FluentAudio package-local 機能です。`so101_voice_frontend.yaml,so101_agent_audio_tools.yaml` を 1 つの FluentAudio system config として読むことは確認できますが、それだけで親 VLAbor profile integration、Agent Runtime / MCP client integration、durable storage、World Station、実 SO101 device / model provisioning が検証済みになったわけではありません。multi-config の代表 runtime launch smoke は後述の combined launch smoke で確認しています。

`src/apps/agent_tools/fa_audio_mcp/test/test_real_owner_graph_smoke.py::test_mcp_tools_call_real_asr_and_audio_window_owner_nodes` は Docker 内で `1 passed` として確認済みです。この smoke は `ros2 run` で実 owner node を起動し、in-process `FastMCP` から real `RosAudioTimelineClient` 経由で `transcribe_audio` / `archive_audio_window` を呼びます。後述の combined launch は別 smoke で確認していますが、この real-owner smoke 単体は installed MCP server transport の検証ではありません。

`src/system/fluent_audio_system/test/integration/test_combined_owner_mcp_launch_smoke.py::test_fluent_audio_system_composes_owner_and_mcp_adapter_configs` は Docker ROS2 環境で `fluent_audio_system` rebuild 後に `python3 -m pytest src/system/fluent_audio_system/test/integration/test_combined_owner_mcp_launch_smoke.py -q` を実行し、`1 passed in 2.39s` として確認済みです。この smoke は `config:=owner.yaml,adapter.yaml` 形式の multi-config list を `ros2 launch fluent_audio_system run.py` に渡し、fake worker / generated params を使いながら、実 `fa_asr` / `fa_asr_node`、`fa_audio_window` / `fa_audio_window_node` owner services と `fa_audio_mcp` / `fa_audio_mcp_server` adapter を同じ launch graph に載せます。起動済み MCP adapter ROS node、real `TranscribeAudio` / `ExportAudioWindow` / `ArchiveAudioWindow` service、deterministic `FLOAT32LE` ASR frame、deterministic `PCM16LE` archive frame、service response の transcript / model ref / window ref、WAV bytes、archive metadata を runtime behavior として検証します。

`src/apps/agent_tools/fa_audio_mcp/test/test_streamable_http_transport_smoke.py::test_installed_server_streamable_http_transport_calls_real_owner_nodes` は、実 `fa_asr_node` と `fa_audio_window_node`、deterministic `FLOAT32LE` ASR frame、deterministic `PCM16LE` archive frame、`ros2 run fa_audio_mcp fa_audio_mcp_server` で起動した installed server を使う smoke です。MCP streamable-http transport で `/mcp` に接続し、`list_tools`、`transcribe_audio`、`archive_audio_window` を呼び、unsupported transcribe scope が MCP error result になることを確認済みです。

Docker cleanup fix 後の `fa_audio_mcp` package 検証では、`colcon test --packages-select fa_audio_mcp --event-handlers console_direct+` が 79 items を収集し、`79 passed in 6.76s`、package は `7.51s` で完了しています。先行して実行した real-owner smoke と streamable smoke の focused pair も `2 passed in 6.41s` として確認済みです。

この combined launch smoke と streamable transport smoke で確認した owner 境界は、`transcribe_audio` の owner が `fa_asr`、`archive_audio_window` / `export_audio_window` の owner が `fa_audio_window` であることです。combined launch smoke は child-side `fluent_audio_system` multi-config launch composition の検証であり、streamable transport smoke は child repo の installed server transport / tool-call smoke です。親 VLAbor profile integration、親 Agent Runtime / MCP client 統合、World Station durable storage、実 SO101 device / model provisioning は検証していません。

### 5.6 録音（WAV）
1. `fa_in`と`fa_record`を起動
2. `record`サービスで開始/停止し、WAVを保存

## 6. 配信 sink（Icecast向け）
`fa_stream`には`audio/frame`をIcecast/Shoutcastへ配信するPython node（`fa_stream_node.py`）を同梱しています。ROS2 node は topic / parameter / `AudioFrame` validation を担当し、`ffmpeg` の起動と書き込みは ROS-free な network streamer backend が担当します。`output_url` は必須で、空のままでは起動失敗します。

`fa_stream` は network sink utility であり、`src/streaming` の jitter buffer / clock drift / PLC などのリアルタイム伝送安定化 node ではありません。

起動例:
```bash
ros2 launch fa_stream fa_stream.launch.py
```
