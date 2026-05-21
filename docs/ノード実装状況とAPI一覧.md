# FluentAudioROS2 ノード実装状況とAPI一覧

## 目的と範囲

この文書は、`src/ai`, `src/apps`, `src/interfaces`, `src/io`, `src/processing`, `src/streaming`, `src/system` 配下の leaf node/package/planned directory を静的に棚卸しした一覧である。根拠は現時点の `package.xml`, `CMakeLists.txt` / `setup.py`, `launch`, `config`, 実装source上の ROS public API に限定する。

これは source/config inventory であり、full build、full test、実ROS graph launch、実デバイス、実モデル、親VLAbor profile integration の完了証明ではない。テスト欄は test directory の有無を示すだけで、runtime API の実装証明としては扱わない。

抽出基準は「`package.xml` を持つpackage directory」または「`fa_` / `fluent_` 名の README-only planned directory」。この基準で leaf entry は 106 件、`find src -name package.xml` は 78 件、内訳は runtime node package 75 件、インターフェース package 1 件（`fa_interfaces`）、支援 package 2 件（`fa_audio_mcp`, `fluent_audio_system`）、計画/未実装 placeholder 28 件である。

外部公開APIとして、ROS topic subscription/publication、ROS service server/client、MCP tool、launch引数、公開config parameter を記載する。ROS 2 Action server/client は source/config inventory 上で公開されている node がないため、現時点では該当なしとして扱う。

## 状態凡例

| 状態 | 意味 |
| --- | --- |
| 実装済み | `package.xml`、実行ファイルまたはnode source、launch/config の public contract が存在する runtime node package。runtime validation 完了は意味しない。 |
| 基盤 | message/service生成、MCP adapter、system launch composition など、audio stream node ではないが public API surface を持つpackage。 |
| 計画/未実装 | README-only planned directory。外部runtime API、package、node、launch、config は未実装。 |

## 全体サマリーテーブル

| 分類 | ノード/パッケージ | 状態 | テスト | 外部API概要 |
| --- | --- | --- | --- | --- |
| AI | `fa_asr` | 実装済み | あり(9 files) | Sub 3; Pub 3; Srv 1 |
| AI | `fa_audio_embedding` | 実装済み | あり(7 files) | Sub 1; Pub 1 |
| AI | `fa_kws` | 実装済み | あり(12 files) | Sub 2; Pub 1 |
| AI | `fa_sed` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| AI | `fa_speaker` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| AI | `fa_turn_detector` | 実装済み | あり(9 files) | Sub 3; Pub 1 |
| AI | `fa_vad` | 実装済み | あり(8 files) | Sub 1; Pub 3 |
| Apps<br>Agent Tools | `fa_audio_mcp` | 基盤 | あり(12 files) | Client 3; MCP export_audio_window, archive_audio_window, transcribe_audio |
| Apps<br>Dialogue | `fa_dialogue` | 実装済み | あり(7 files) | Sub 3; Pub 1 |
| Apps<br>Safety | `fa_safety_policy` | 計画/未実装 | あり(4 files) | 未実装。外部runtime APIなし。 |
| Apps<br>Voice Command | `fa_voice_command_router` | 実装済み | あり(6 files) | Sub 1; Pub 2; Srv 3; Client 1 |
| Interfaces | `fa_interfaces` | 基盤 | あり(5 files) | msg 22 / srv 8 を生成。 |
| IO<br>Sinks | `fa_file_out` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| IO<br>Sinks | `fa_network_out` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| IO<br>Sinks | `fa_out` | 実装済み | あり(9 files) | Sub 1; Pub 1; Srv 1 |
| IO<br>Sources | `fa_file_in` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| IO<br>Sources | `fa_in` | 実装済み | あり(9 files) | Pub 2; Srv 2 |
| IO<br>Sources | `fa_network_in` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| IO<br>Utilities | `fa_record` | 実装済み | あり(6 files) | Sub 1; Srv 1 |
| IO<br>Utilities | `fa_stream` | 実装済み | あり(6 files) | Sub 1 |
| Processing<br>Analysis and Feature | `fa_cqt` | 実装済み | あり(5 files) | Sub 1; Pub 1 |
| Processing<br>Analysis and Feature | `fa_log_mel` | 実装済み | あり(5 files) | Sub 1; Pub 1 |
| Processing<br>Analysis and Feature | `fa_loudness` | 実装済み | あり(5 files) | Sub 1; Pub 1 |
| Processing<br>Analysis and Feature | `fa_mfcc` | 実装済み | あり(5 files) | Sub 1; Pub 1 |
| Processing<br>Analysis and Feature | `fa_onset` | 実装済み | あり(5 files) | Sub 1; Pub 1 |
| Processing<br>Analysis and Feature | `fa_pitch` | 実装済み | あり(5 files) | Sub 1; Pub 1 |
| Processing<br>Analysis and Feature | `fa_stft` | 実装済み | あり(5 files) | Sub 1; Pub 1 |
| Processing<br>Analysis and Feature | `fa_tempo` | 実装済み | あり(5 files) | Sub 1; Pub 1 |
| Processing<br>Correction and Noise | `fa_aec_linear` | 実装済み | あり(7 files) | Sub 2; Pub 2 |
| Processing<br>Correction and Noise | `fa_aec_nn` | 実装済み | あり(6 files) | Sub 1; Pub 2 |
| Processing<br>Correction and Noise | `fa_dc_offset_removal` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Correction and Noise | `fa_debreath` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Correction and Noise | `fa_declick` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Correction and Noise | `fa_declip` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Correction and Noise | `fa_denoise` | 実装済み | あり(8 files) | Sub 1; Pub 2 |
| Processing<br>Correction and Noise | `fa_dereverb` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Correction and Noise | `fa_hum` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Correction and Noise | `fa_wind` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Dynamics | `fa_agc` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Dynamics | `fa_compressor` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Dynamics | `fa_expander` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Dynamics | `fa_gain` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Dynamics | `fa_limiter` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Dynamics | `fa_noise_gate` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Dynamics | `fa_normalize` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Format | `fa_bit_depth` | 実装済み | あり(6 files) | Sub 1; Pub 2 |
| Processing<br>Format | `fa_channel_convert` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Format | `fa_decode` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Format | `fa_encode` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Format | `fa_format` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Format | `fa_interleave` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Format | `fa_resample` | 実装済み | あり(9 files) | Sub 2; Pub 3 |
| Processing<br>Format | `fa_sample_format` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Frequency | `fa_band_pass` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Frequency | `fa_deesser` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Frequency | `fa_eq` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Frequency | `fa_filter` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Frequency | `fa_high_pass` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Frequency | `fa_low_pass` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Frequency | `fa_notch` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Processing<br>Frequency | `fa_spectral_subtraction` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Frequency | `fa_wiener` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Generation | `fa_music_source_separation` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Generation | `fa_neural_codec` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Generation | `fa_neural_vocoder` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Generation | `fa_speech_enhancement` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Generation | `fa_speech_separation` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Generation | `fa_speech_translation` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Generation | `fa_super_resolution` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Generation | `fa_tts` | 実装済み | あり(5 files) | Pub 1; Srv 1 |
| Processing<br>Generation | `fa_voice_conversion` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Routing and Mixing | `fa_bus_router` | 実装済み | あり(3 files) | Sub 1; Pub 2 |
| Processing<br>Routing and Mixing | `fa_ducking` | 実装済み | あり(5 files) | Sub 2; Pub 2 |
| Processing<br>Routing and Mixing | `fa_loopback` | 実装済み | あり(3 files) | Sub 1; Pub 2 |
| Processing<br>Routing and Mixing | `fa_mix` | 実装済み | あり(8 files) | Sub 1; Pub 2 |
| Processing<br>Routing and Mixing | `fa_monitor_mix` | 実装済み | あり(4 files) | Sub 1; Pub 2 |
| Processing<br>Routing and Mixing | `fa_patchbay` | 実装済み | あり(3 files) | Sub 1; Pub 2 |
| Processing<br>Routing and Mixing | `fa_sidechain` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Spatial and Channel | `fa_ambisonics` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Spatial and Channel | `fa_beamforming` | 実装済み | あり(4 files) | Sub 1; Pub 2 |
| Processing<br>Spatial and Channel | `fa_binaural` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Spatial and Channel | `fa_downmix` | 実装済み | あり(4 files) | Sub 1; Pub 2 |
| Processing<br>Spatial and Channel | `fa_pan` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Spatial and Channel | `fa_source_separation` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Spatial and Channel | `fa_stereo_widening` | 実装済み | あり(4 files) | Sub 1; Pub 2 |
| Processing<br>Spatial and Channel | `fa_upmix` | 実装済み | あり(4 files) | Sub 1; Pub 2 |
| Processing<br>Temporal | `fa_crossfade` | 実装済み | あり(5 files) | Sub 2; Pub 2 |
| Processing<br>Temporal | `fa_delay` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Temporal | `fa_echo` | 実装済み | あり(6 files) | Sub 1; Pub 2 |
| Processing<br>Temporal | `fa_fade` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Temporal | `fa_pitch_shift` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Temporal | `fa_reverb` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Temporal | `fa_silence_removal` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Temporal | `fa_time_stretch` | 計画/未実装 | あり(3 files) | 未実装。外部runtime APIなし。 |
| Processing<br>Temporal | `fa_trim` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Processing<br>Temporal | `fa_window` | 実装済み | あり(5 files) | Sub 1; Pub 2 |
| Streaming | `fa_audio_window` | 実装済み | あり(3 files) | Sub 1; Srv 2 |
| Streaming | `fa_chunk_overlap` | 実装済み | あり(6 files) | Sub 1; Pub 2 |
| Streaming | `fa_clock_drift` | 実装済み | あり(6 files) | Sub 1; Pub 2 |
| Streaming | `fa_frame_buffer` | 実装済み | あり(7 files) | Sub 1; Pub 2 |
| Streaming | `fa_jitter_buffer` | 実装済み | あり(6 files) | Sub 1; Pub 2 |
| Streaming | `fa_latency_compensation` | 実装済み | あり(6 files) | Sub 1; Pub 2 |
| Streaming | `fa_overlap_add` | 実装済み | あり(6 files) | Sub 1; Pub 2 |
| Streaming | `fa_packet_loss_concealment` | 実装済み | あり(6 files) | Sub 1; Pub 2 |
| Streaming | `fa_time_alignment` | 実装済み | あり(6 files) | Sub 1; Pub 2 |
| System | `fluent_audio_system` | 基盤 | あり(21 files) | system YAMLをlaunch graphへ展開。CLI `list_required_packages`。 |

## AI

### `fa_asr`
- 状態: 実装済み。
- 根拠path: `src/ai/fa_asr/package.xml`, `src/ai/fa_asr/CMakeLists.txt`, `src/ai/fa_asr/config/default.yaml`, `src/ai/fa_asr/launch/fa_asr.launch.py`, `src/ai/fa_asr/fa_asr_py/asr_node.py`, `src/ai/fa_asr/test`。
- 実行ファイル/ROS node: exec `fa_asr_node`; node `fa_asr`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `audio_topic` default `audio/frame` / `fa_interfaces/msg/AudioFrame`; `turn_context_topic` default `conversation/turn_context` / `fa_interfaces/msg/TurnContext`; `control.speech_control.topic` default `voice/vad_state` / `fa_interfaces/msg/VadState`
- Publishers: `asr_result_topic` default `voice/asr/result` / `fa_interfaces/msg/AsrResult`; `asr_state_topic` default `voice/asr/state` / `fa_interfaces/msg/AsrState`; `asr_event_topic` default `voice/asr/event` / `fa_interfaces/msg/AsrEvent`
- Services: `transcribe_service_name` default `transcribe_audio` / `fa_interfaces/srv/TranscribeAudio`
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `audio_topic`=`audio/frame`; `turn_context_topic`=`conversation/turn_context`; `asr_result_topic`=`voice/asr/result`; `asr_state_topic`=`voice/asr/state`; `asr_event_topic`=`voice/asr/event`; `transcribe_service_name`=`transcribe_audio`
  - identity/scope: `expected_source_id`=`""`; `expected_stream_id`=`""`
  - control: `control.default_enabled`=`false`; `control.inputs`=`["speech_control"]`; `control.speech_control.action`=`topic`; `control.speech_control.topic`=`voice/vad_state`; `control.speech_control.msg_type`=`fa_interfaces/msg/VadState`; `control.speech_control.source_id`=`""`; `control.speech_control.stream_id`=`""`; `control.speech_control.active_field`=`is_speech`; `control.speech_control.start_field`=`start`; `control.speech_control.end_field`=`end`; `control.speech_control.open_on`=`start_or_active_rising`; `control.speech_control.close_on`=`end_or_active_falling`; `control.speech_control.submit_on_close`=`true`; `control.speech_control.pre_roll_ms`=`0.0`; `control.speech_control.post_roll_ms`=`0.0`
  - backend/model/external: `backend.name`=`""`; `backend.kind`=`asr`; `backend.model`=`""`; `backend.command`=`""`; `backend.model_path`=`""`; `backend.model_version`=`""`; `backend.model_revision`=`""`; `backend.openai_realtime.api_key_env`=`""`; `backend.openai_transcriptions.api_key_env`=`""`; `backend.language`=`ja`; `backend.timeout_sec`=`120.0`; `backend.working_directory`=`""`; `backend.args`=`[]`; `backend.health_args`=`[]`; `backend.output_text_path`=`""`; `backend.result_format`=`""`; `backend.sample_rate_hz`=`16000`; `backend.channels`=`1`; `backend.chunk_size_samples`=`1600`; `backend.chunk_ms`=`0`; `backend.emit_partial`=`true`; `backend.max_partial_interval_ms`=`300`
  - format: `target_sample_rate`=`16000`
  - timeline: `timeline.retention_sec`=`1800.0`; `timeline.timestamp_alignment_tolerance_ms`=`1.0`; `timeline.clock`=`media`; `timeline.window_id`=`fa_asr_rolling_asr_window`; `timeline.window_epoch`=`0`
  - trace/observability: `trace.enabled`=`false`; `trace.path`=`""`; `observability.qos.depth`=`50`; `observability.qos.reliable`=`true`
  - QoS: `audio.qos.depth`=`20`; `audio.qos.reliable`=`false`; `control.speech_control.qos.depth`=`50`; `control.speech_control.qos.reliable`=`false`; `turn_context.qos.depth`=`10`; `turn_context.qos.reliable`=`true`; `result.qos.depth`=`10`; `result.qos.reliable`=`true`
  - other public: `min_audio_sec`=`0.3`; `silence_timeout_sec`=`10.0`; `finalize_on_context_inactive`=`true`; `workspace_dir`=`/tmp/fa_asr`; `cleanup_audio_files`=`true`

### `fa_audio_embedding`
- 状態: 実装済み。
- 根拠path: `src/ai/fa_audio_embedding/package.xml`, `src/ai/fa_audio_embedding/CMakeLists.txt`, `src/ai/fa_audio_embedding/config/default.yaml`, `src/ai/fa_audio_embedding/launch/fa_audio_embedding.launch.py`, `src/ai/fa_audio_embedding/fa_audio_embedding_py/audio_embedding_node.py`, `src/ai/fa_audio_embedding/test`。
- 実行ファイル/ROS node: exec `fa_audio_embedding_node`; node `fa_audio_embedding`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/embedding/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/embedding/frame` / `fa_interfaces/msg/AudioEmbeddingFrame`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/embedding/input`; `output_topic`=`audio/embedding/frame`
  - identity/scope: `expected_source_id`=`""`; `expected_stream_id`=`""`
  - backend/model/external: `backend.name`=`""`; `backend.command`=`""`; `backend.model_id`=`""`; `backend.model_path`=`""`; `backend.args`=`[]`; `backend.payload_encoding`=`float32le_raw`; `backend.timeout_sec`=`30.0`; `backend.workspace_dir`=`/tmp/fa_audio_embedding`; `backend.cleanup_audio_files`=`true`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`

### `fa_kws`
- 状態: 実装済み。
- 根拠path: `src/ai/fa_kws/package.xml`, `src/ai/fa_kws/CMakeLists.txt`, `src/ai/fa_kws/config/default.yaml`, `src/ai/fa_kws/launch/fa_kws.launch.py`, `src/ai/fa_kws/src/fa_kws_node.cpp`, `src/ai/fa_kws/test`。
- 実行ファイル/ROS node: exec `fa_kws_node`, `fa_kws_wav_tool`, `sherpa_onnx_kws_worker`; node `fa_kws`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `audio_topic` default `audio/frame` / `fa_interfaces/msg/AudioFrame`; `vad_topic` default `voice/vad_state` / `fa_interfaces/msg/VadState`
- Publishers: `output_topic` default `voice/wake_word` / `fa_interfaces/msg/WakeWordResult`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `audio_topic`=`audio/frame`; `vad_topic`=`voice/vad_state`; `output_topic`=`voice/wake_word`
  - identity/scope: `expected_stream_id`=`audio/raw/mic`; `expected_source_id`=`""`
  - backend/model/external: `backend.name`=`""`; `model.encoder`=`""`; `model.decoder`=`""`; `model.joiner`=`""`; `model.tokens`=`""`; `kws.keywords_file`=`""`; `model.num_threads`=`4`; `backend.execution_provider`=`""`; `backend.command`=`""`; `backend.args`=`[]`; `backend.health_args`=`[]`; `backend.timeout_sec`=`5.0`; `backend.workspace_dir`=`/tmp/fluent_audio_fa_kws`; `backend.cleanup_audio_files`=`true`; `kws.max_active_paths`=`32`; `kws.num_trailing_blanks`=`1`; `kws.keywords_score`=`1.0`; `kws.keywords_threshold`=`0.25`
  - format: `target_sample_rate`=`16000`
  - QoS: `audio.qos.depth`=`10`; `audio.qos.reliable`=`false`; `vad.qos.depth`=`20`; `vad.qos.reliable`=`false`; `output.qos.depth`=`10`; `output.qos.reliable`=`false`
  - other public: `vad.probability_gate`=`0.35`; `vad.max_age_ms`=`1000`

### `fa_sed`
- 状態: 計画/未実装。
- 根拠path: `src/ai/fa_sed/README.md`, `src/ai/fa_sed/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_speaker`
- 状態: 計画/未実装。
- 根拠path: `src/ai/fa_speaker/README.md`, `src/ai/fa_speaker/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_turn_detector`
- 状態: 実装済み。
- 根拠path: `src/ai/fa_turn_detector/package.xml`, `src/ai/fa_turn_detector/CMakeLists.txt`, `src/ai/fa_turn_detector/config/default.yaml`, `src/ai/fa_turn_detector/launch/fa_turn_detector.launch.py`, `src/ai/fa_turn_detector/fa_turn_detector_py/turn_detector_node.py`, `src/ai/fa_turn_detector/test`。
- 実行ファイル/ROS node: exec `fa_turn_detector_node`, `smart_turn_onnx_worker`; node `fa_turn_detector`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `audio_topic` default `audio/frame` / `fa_interfaces/msg/AudioFrame`; `vad_topic` default `voice/vad_state` / `fa_interfaces/msg/VadState`; `turn_context_topic` default `conversation/turn_context` / `fa_interfaces/msg/TurnContext`
- Publishers: `output_topic` default `voice/turn_end` / `fa_interfaces/msg/TurnEnd`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `audio_topic`=`audio/frame`; `vad_topic`=`voice/vad_state`; `turn_context_topic`=`conversation/turn_context`; `output_topic`=`voice/turn_end`
  - identity/scope: `expected_stream_id`=`audio/raw/mic`; `expected_source_id`=`""`
  - backend/model/external: `backend.name`=`""`; `backend.model_path`=`""`; `backend.execution_provider`=`""`; `backend.command`=`""`; `backend.args`=`[]`; `backend.health_args`=`[]`; `backend.timeout_sec`=`5.0`; `backend.workspace_dir`=`/tmp/fluent_audio_fa_turn_detector`; `backend.cleanup_audio_files`=`true`; `backend.threshold`=`0.5`
  - QoS: `audio.qos.depth`=`10`; `audio.qos.reliable`=`false`; `vad.qos.depth`=`10`; `vad.qos.reliable`=`false`; `turn_context.qos.depth`=`10`; `turn_context.qos.reliable`=`true`; `output.qos.depth`=`10`; `output.qos.reliable`=`true`

### `fa_vad`
- 状態: 実装済み。
- 根拠path: `src/ai/fa_vad/package.xml`, `src/ai/fa_vad/CMakeLists.txt`, `src/ai/fa_vad/config/default.yaml`, `src/ai/fa_vad/launch/fa_vad.launch.py`, `src/ai/fa_vad/fa_vad_py/vad_node.py`, `src/ai/fa_vad/test`。
- 実行ファイル/ROS node: exec `fa_vad_node`, `silero_vad_worker`; node `fa_vad`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/frame` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/vad` / `std_msgs/msg/Bool`; `vad_state_topic` default `voice/vad_state` / `fa_interfaces/msg/VadState`; `probability_topic` default `audio/vad/probability` / `std_msgs/msg/Float32`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/frame`; `output_topic`=`audio/vad`; `vad_state_topic`=`voice/vad_state`; `probability_topic`=`audio/vad/probability`
  - identity/scope: `input_stream_id`=`audio/raw/mic`; `expected_source_id`=`""`
  - backend/model/external: `backend.name`=`""`; `backend.frame_ms`=`20`; `backend.window_samples`=`512`; `backend.history_buffer_ms`=`200`; `backend.model_path`=`""`; `backend.execution_provider`=`""`; `backend.command`=`""`; `backend.args`=`["--audio", "{audio}", "--model", "{model}", "--provider", "{provider}", "--sample-rate", "{sample_rate}", "--window-samples", "{window_samples}"]`; `backend.timeout_sec`=`1.0`; `backend.workspace_dir`=`/tmp/fluent_audio/fa_vad`; `backend.cleanup_audio_files`=`true`
  - format: `target_sample_rate`=`16000`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`

## Apps / Agent Tools / Dialogue / Voice Command

### `fa_audio_mcp`
- 状態: 基盤。
- 根拠path: `src/apps/agent_tools/fa_audio_mcp/package.xml`, `src/apps/agent_tools/fa_audio_mcp/setup.py`, `src/apps/agent_tools/fa_audio_mcp/config/default.yaml`, `src/apps/agent_tools/fa_audio_mcp/fa_audio_mcp/config.py`, `src/apps/agent_tools/fa_audio_mcp/fa_audio_mcp/server.py`, `src/apps/agent_tools/fa_audio_mcp/test`。
- 実行ファイル/ROS node: exec `fa_audio_mcp_server`; node `fa_audio_mcp_server`。
- Launch: なし。
- Subscriptions: なし。
- Publishers: なし。
- Services: なし。
- Clients / MCP tools: `FLUENT_AUDIO_EXPORT_AUDIO_WINDOW_SERVICE` default `export_audio_window` / `fa_interfaces/srv/ExportAudioWindow`; `FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE` default `archive_audio_window` / `fa_interfaces/srv/ArchiveAudioWindow`; `FLUENT_AUDIO_TRANSCRIBE_AUDIO_SERVICE` default `transcribe_audio` / `fa_interfaces/srv/TranscribeAudio`; `export_audio_window`; `archive_audio_window`; `transcribe_audio`
- Public config parameters:
  - 環境変数: `FLUENT_AUDIO_MCP_TRANSPORT` default `stdio` (`stdio` / `sse` / `streamable-http`), `FLUENT_AUDIO_MCP_HOST` default `0.0.0.0`, `FLUENT_AUDIO_MCP_PORT` default `9110`, `FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC` default `10.0`。
  - サービス名環境変数: `FLUENT_AUDIO_EXPORT_AUDIO_WINDOW_SERVICE` default `export_audio_window`, `FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE` default `archive_audio_window`, `FLUENT_AUDIO_TRANSCRIBE_AUDIO_SERVICE` default `transcribe_audio`。
  - scope環境変数: `FLUENT_AUDIO_EXPORT_SCOPE_*`, `FLUENT_AUDIO_ARCHIVE_SCOPE_*`, `FLUENT_AUDIO_TRANSCRIBE_SCOPE_*`, `FLUENT_AUDIO_*_DEFAULT_SCOPE`; time markerは `FLUENT_AUDIO_TIME_MARKERS`。

### `fa_dialogue`
- 状態: 実装済み。
- 根拠path: `src/apps/dialogue/fa_dialogue/package.xml`, `src/apps/dialogue/fa_dialogue/CMakeLists.txt`, `src/apps/dialogue/fa_dialogue/config/default.yaml`, `src/apps/dialogue/fa_dialogue/launch/fa_dialogue.launch.py`, `src/apps/dialogue/fa_dialogue/fa_dialogue_py/dialogue_node.py`, `src/apps/dialogue/fa_dialogue/test`。
- 実行ファイル/ROS node: exec `fa_dialogue_node`; node `fa_dialogue`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `wake_word_topic` default `voice/wake_word` / `fa_interfaces/msg/WakeWordResult`; `asr_result_topic` default `voice/asr/result` / `fa_interfaces/msg/AsrResult`; `turn_end_topic` default `voice/turn_end` / `fa_interfaces/msg/TurnEnd`
- Publishers: `turn_context_topic` default `conversation/turn_context` / `fa_interfaces/msg/TurnContext`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `wake_word_topic`=`voice/wake_word`; `asr_result_topic`=`voice/asr/result`; `turn_end_topic`=`voice/turn_end`; `turn_context_topic`=`conversation/turn_context`
  - QoS: `wake.qos.depth`=`10`; `wake.qos.reliable`=`true`; `asr.qos.depth`=`10`; `asr.qos.reliable`=`true`; `turn_end.qos.depth`=`10`; `turn_end.qos.reliable`=`true`; `turn_context.qos.depth`=`10`; `turn_context.qos.reliable`=`true`
  - other public: `session_prefix`=`dialogue-session-`; `wake.max_age_ms`=`1500`; `wake.allow_zero_stamp`=`false`

### `fa_safety_policy`
- 状態: 計画/未実装。
- 根拠path: `src/apps/safety/fa_safety_policy/README.md`, `src/apps/safety/fa_safety_policy/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_voice_command_router`
- 状態: 実装済み。
- 根拠path: `src/apps/voice_command/fa_voice_command_router/package.xml`, `src/apps/voice_command/fa_voice_command_router/CMakeLists.txt`, `src/apps/voice_command/fa_voice_command_router/config/default.yaml`, `src/apps/voice_command/fa_voice_command_router/launch/fa_voice_command_router.launch.py`, `src/apps/voice_command/fa_voice_command_router/fa_voice_command_router_py/router_node.py`, `src/apps/voice_command/fa_voice_command_router/test`。
- 実行ファイル/ROS node: exec `fa_voice_command_router_node`; node `fa_voice_command_router`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `command_topic` default `voice/command` / `std_msgs/msg/String`
- Publishers: `state_topic` default `voice/router/state` / `std_msgs/msg/String`; `output_stop_topic` default `audio/output/stop` / `std_msgs/msg/Empty`
- Services: `start` / `std_srvs/srv/Trigger`; `stop` / `std_srvs/srv/Trigger`; `status` / `std_srvs/srv/Trigger`
- Clients / MCP tools: `tts_service` default `speak` / `fa_interfaces/srv/Speak`
- Public config parameters:
  - topic/service: `command_topic`=`voice/command`; `state_topic`=`voice/router/state`; `tts_service`=`speak`; `output_stop_topic`=`audio/output/stop`
  - other public: `active`=`false`; `mode`=`standby`; `allowed_modes`=`["standby", "command", "dictation", "mute"]`; `announce_tts`=`false`; `stop_output_on_stop`=`true`

## Interfaces

### `fa_interfaces`
- 状態: 基盤。
- 根拠path: `src/interfaces/fa_interfaces/package.xml`, `src/interfaces/fa_interfaces/CMakeLists.txt`, `src/interfaces/fa_interfaces/test`。
- 実行ファイル/ROS node: なし。インターフェース生成package。
- Messages: `AsrResult.msg`, `AudioClipRef.msg`, `AudioEmbeddingFrame.msg`, `AudioFrame.msg`, `AudioModelRef.msg`, `AudioWindowRef.msg`, `CqtFrame.msg`, `EncodedAudioChunk.msg`, `LogMelFrame.msg`, `LoudnessFrame.msg`, `MfccFrame.msg`, `OnsetFrame.msg`, `PitchFrame.msg`, `PlaybackDone.msg`, `ResolvedTimeRange.msg`, `StftFrame.msg`, `TempoFrame.msg`, `TranscriptSegment.msg`, `TurnContext.msg`, `TurnEnd.msg`, `VadState.msg`, `WakeWordResult.msg`。
- Services: `ArchiveAudioWindow.srv`, `ExportAudioWindow.srv`, `ListDevices.srv`, `PlaybackControl.srv`, `Record.srv`, `Speak.srv`, `SwitchDevice.srv`, `TranscribeAudio.srv`。
- Public config parameters: なし。

## IO Sources/Sinks/Utilities

### `fa_file_in`
- 状態: 計画/未実装。
- 根拠path: `src/io/sources/fa_file_in/README.md`, `src/io/sources/fa_file_in/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_in`
- 状態: 実装済み。
- 根拠path: `src/io/sources/fa_in/package.xml`, `src/io/sources/fa_in/CMakeLists.txt`, `src/io/sources/fa_in/config/default.yaml`, `src/io/sources/fa_in/launch/fa_in.launch.py`, `src/io/sources/fa_in/src/fa_in_node.cpp`, `src/io/sources/fa_in/test`。
- 実行ファイル/ROS node: exec `fa_in_node`; node `fa_in`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: なし。
- Publishers: `output_topic` default `fa_in/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: `list_devices` / `fa_interfaces/srv/ListDevices`; `switch_device` / `fa_interfaces/srv/SwitchDevice`
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `output_topic`=`fa_in/output`
  - identity/scope: `audio.stream_id`=`audio/raw/mic`
  - backend/model/external: `backend.name`=`alsa_capture`
  - format: `audio.sample_rate`=`48000`; `audio.channels`=`1`; `audio.bit_depth`=`16`; `audio.encoding`=`PCM16LE`; `audio.layout`=`interleaved`
  - QoS: `audio.qos.depth`=`10`; `audio.qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`
  - other public: `audio.device_selector.mode`=`id`; `audio.device_selector.identifier`=`""`; `audio.device_selector.index`=`-1`; `audio.chunk_ms`=`20`; `startup.required_subscribers`=`0`; `startup.subscriber_wait_timeout_ms`=`0`
  - backend-specific declared params: `file.path`, `endpoint.uri`, `transport.identity`, `audio.source_id`, `playback.loop`, `network.max_packet_bytes`, `polling.period_ms`, `network.source_timeout_ms`

### `fa_network_in`
- 状態: 計画/未実装。
- 根拠path: `src/io/sources/fa_network_in/README.md`, `src/io/sources/fa_network_in/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_file_out`
- 状態: 計画/未実装。
- 根拠path: `src/io/sinks/fa_file_out/README.md`, `src/io/sinks/fa_file_out/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_network_out`
- 状態: 計画/未実装。
- 根拠path: `src/io/sinks/fa_network_out/README.md`, `src/io/sinks/fa_network_out/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_out`
- 状態: 実装済み。
- 根拠path: `src/io/sinks/fa_out/package.xml`, `src/io/sinks/fa_out/CMakeLists.txt`, `src/io/sinks/fa_out/config/default.yaml`, `src/io/sinks/fa_out/launch/fa_out.launch.py`, `src/io/sinks/fa_out/src/fa_out_node.cpp`, `src/io/sinks/fa_out/test`。
- 実行ファイル/ROS node: exec `fa_out_node`; node `fa_out`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_out/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `playback_done_topic` default `fa_out/playback_done` / `fa_interfaces/msg/PlaybackDone`
- Services: `playback_control_service` default `fa_out/playback_control` / `fa_interfaces/srv/PlaybackControl`
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_out/input`; `playback_done_topic`=`fa_out/playback_done`; `playback_control_service`=`fa_out/playback_control`
  - identity/scope: `input_stream_id`=`audio/playback/main`
  - backend/model/external: `backend.name`=`alsa_playback`
  - format: `audio.encoding`=`PCM16LE`; `audio.sample_rate`=`48000`; `audio.channels`=`1`; `audio.bit_depth`=`16`
  - QoS: `audio.qos.depth`=`10`; `audio.qos.reliable`=`true`; `lifecycle.qos.depth`=`10`; `lifecycle.qos.reliable`=`true`
  - other public: `audio.device_id`=`""`; `audio.chunk_duration_ms`=`30`; `audio.alsa.buffer_frames`=`16384`; `audio.alsa.period_frames`=`4096`; `queue.max_frames`=`32`
  - backend-specific declared params: `file.path`, `overwrite.enabled`, `endpoint.uri`, `transport.identity`, `network.max_packet_bytes`

### `fa_record`
- 状態: 実装済み。
- 根拠path: `src/io/utilities/fa_record/package.xml`, `src/io/utilities/fa_record/CMakeLists.txt`, `src/io/utilities/fa_record/config/default.yaml`, `src/io/utilities/fa_record/launch/fa_record.launch.py`, `src/io/utilities/fa_record/src/fa_record_node.cpp`, `src/io/utilities/fa_record/test`。
- 実行ファイル/ROS node: exec `fa_record_node`; node `fa_record`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/frame` / `fa_interfaces/msg/AudioFrame`
- Publishers: なし。
- Services: `record` / `fa_interfaces/srv/Record`
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/frame`
  - QoS: `input.qos.depth`=`10`; `input.qos.reliable`=`true`

### `fa_stream`
- 状態: 実装済み。
- 根拠path: `src/io/utilities/fa_stream/package.xml`, `src/io/utilities/fa_stream/CMakeLists.txt`, `src/io/utilities/fa_stream/config/default.yaml`, `src/io/utilities/fa_stream/launch/fa_stream.launch.py`, `src/io/utilities/fa_stream/scripts/fa_stream_node.py`, `src/io/utilities/fa_stream/test`。
- 実行ファイル/ROS node: exec `fa_stream_node.py`; node `fa_stream`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/frame` / `fa_interfaces/msg/AudioFrame`
- Publishers: なし。
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/frame`
  - backend/model/external: `ffmpeg_path`=`ffmpeg`; `output_url`=`""`; `audio_codec`=`libmp3lame`; `bitrate`=`128k`; `container_format`=`mp3`; `content_type`=`audio/mpeg`

## Processing

### Processing: format

### `fa_bit_depth`
- 状態: 実装済み。
- 根拠path: `src/processing/format/fa_bit_depth/package.xml`, `src/processing/format/fa_bit_depth/CMakeLists.txt`, `src/processing/format/fa_bit_depth/config/default.yaml`, `src/processing/format/fa_bit_depth/launch/fa_bit_depth.launch.py`, `src/processing/format/fa_bit_depth/src/fa_bit_depth_node.cpp`, `src/processing/format/fa_bit_depth/test`。
- 実行ファイル/ROS node: exec `fa_bit_depth_node`; node `fa_bit_depth`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_bit_depth/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_bit_depth/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_bit_depth/input`; `output_topic`=`fa_bit_depth/output`
  - identity/scope: `input_stream_id`=`audio/raw/mic`; `output.stream_id`=`audio/bit_depth/mic`
  - format: `input.encoding`=`PCM16LE`; `input.bit_depth`=`16`; `output.encoding`=`PCM32LE`; `output.bit_depth`=`32`; `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_channel_convert`
- 状態: 実装済み。
- 根拠path: `src/processing/format/fa_channel_convert/package.xml`, `src/processing/format/fa_channel_convert/CMakeLists.txt`, `src/processing/format/fa_channel_convert/config/default.yaml`, `src/processing/format/fa_channel_convert/launch/fa_channel_convert.launch.py`, `src/processing/format/fa_channel_convert/src/fa_channel_convert_node.cpp`, `src/processing/format/fa_channel_convert/test`。
- 実行ファイル/ROS node: exec `fa_channel_convert_node`; node `fa_channel_convert`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_channel_convert/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_channel_convert/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_channel_convert/input`; `output_topic`=`fa_channel_convert/output`
  - identity/scope: `input_stream_id`=`audio/sample_format/mic`; `output.stream_id`=`audio/channel_converted/mic`
  - format: `input.channels`=`1`; `output.channels`=`2`; `expected.sample_rate`=`16000`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `conversion.mode`=`mono_to_stereo_duplicate`

### `fa_decode`
- 状態: 実装済み。
- 根拠path: `src/processing/format/fa_decode/package.xml`, `src/processing/format/fa_decode/CMakeLists.txt`, `src/processing/format/fa_decode/config/default.yaml`, `src/processing/format/fa_decode/launch/fa_decode.launch.py`, `src/processing/format/fa_decode/src/fa_decode_node.cpp`, `src/processing/format/fa_decode/test`。
- 実行ファイル/ROS node: exec `fa_decode_node`; node `fa_decode`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/encoded/mic` / `fa_interfaces/msg/EncodedAudioChunk`
- Publishers: `output_topic` default `audio/pcm16/mic` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/encoded/mic`; `output_topic`=`audio/pcm16/mic`
  - identity/scope: `input_stream_id`=`audio/encoded/mic/opus`; `output.stream_id`=`audio/decoded/mic/pcm16`
  - backend/model/external: `backend.name`=`external_codec_decoder`; `backend.command.executable`=`""`; `backend.command.arguments`=`[]`; `backend.command.timeout_ms`=`3000`; `backend.command.max_output_bytes`=`1048576`
  - format: `input.codec`=`opus`; `input.container`=`ogg`; `input.payload_format`=`ogg_page`; `input.sample_rate`=`16000`; `input.channels`=`1`; `output.sample_rate`=`16000`; `output.channels`=`1`; `output.encoding`=`PCM16LE`; `output.bit_depth`=`16`; `output.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_encode`
- 状態: 実装済み。
- 根拠path: `src/processing/format/fa_encode/package.xml`, `src/processing/format/fa_encode/CMakeLists.txt`, `src/processing/format/fa_encode/config/default.yaml`, `src/processing/format/fa_encode/launch/fa_encode.launch.py`, `src/processing/format/fa_encode/src/fa_encode_node.cpp`, `src/processing/format/fa_encode/test`。
- 実行ファイル/ROS node: exec `fa_encode_node`; node `fa_encode`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/pcm16/mic` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/encoded/mic` / `fa_interfaces/msg/EncodedAudioChunk`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/pcm16/mic`; `output_topic`=`audio/encoded/mic`
  - identity/scope: `input_stream_id`=`audio/pcm16/mic/stream`; `output.stream_id`=`audio/encoded/mic/opus`
  - backend/model/external: `backend.name`=`external_codec_encoder`; `backend.command.executable`=`""`; `backend.command.arguments`=`[]`; `backend.command.timeout_ms`=`3000`; `backend.command.max_output_bytes`=`1048576`
  - format: `input.sample_rate`=`16000`; `input.channels`=`1`; `input.encoding`=`PCM16LE`; `input.bit_depth`=`16`; `input.layout`=`interleaved`; `output.codec`=`opus`; `output.container`=`ogg`; `output.payload_format`=`ogg_page`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_format`
- 状態: 計画/未実装。
- 根拠path: `src/processing/format/fa_format/README.md`, `src/processing/format/fa_format/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_interleave`
- 状態: 実装済み。
- 根拠path: `src/processing/format/fa_interleave/package.xml`, `src/processing/format/fa_interleave/CMakeLists.txt`, `src/processing/format/fa_interleave/config/default.yaml`, `src/processing/format/fa_interleave/launch/fa_interleave.launch.py`, `src/processing/format/fa_interleave/src/fa_interleave_node.cpp`, `src/processing/format/fa_interleave/test`。
- 実行ファイル/ROS node: exec `fa_interleave_node`; node `fa_interleave`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_interleave/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_interleave/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_interleave/input`; `output_topic`=`fa_interleave/output`
  - identity/scope: `input_stream_id`=`audio/sample_format/mic`; `output.stream_id`=`audio/layout_reordered/mic`
  - format: `input.layout`=`interleaved`; `output.layout`=`planar`; `expected.sample_rate`=`16000`; `expected.channels`=`2`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_resample`
- 状態: 実装済み。
- 根拠path: `src/processing/format/fa_resample/package.xml`, `src/processing/format/fa_resample/CMakeLists.txt`, `src/processing/format/fa_resample/config/default.yaml`, `src/processing/format/fa_resample/launch/fa_resample.launch.py`, `src/processing/format/fa_resample/src/fa_resample_node.cpp`, `src/processing/format/fa_resample/test`。
- 実行ファイル/ROS node: exec `fa_resample_node`; node `fa_resample`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `mic.input_topic` default `audio/frame` / `fa_interfaces/msg/AudioFrame`; `ref.input_topic` default `audio/output/frame` / `fa_interfaces/msg/AudioFrame`
- Publishers: `mic.output_topic` default `audio/resample16k/mic` / `fa_interfaces/msg/AudioFrame`; `ref.output_topic` default `audio/resample16k/ref` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `mic.input_topic`=`audio/frame`; `mic.output_topic`=`audio/resample16k/mic`; `ref.input_topic`=`audio/output/frame`; `ref.output_topic`=`audio/resample16k/ref`
  - identity/scope: `mic.input_stream_id`=`audio/float32/mic`; `mic.output.stream_id`=`audio/preprocessed/mono16k`; `ref.input_stream_id`=`audio/playback/main`; `ref.output.stream_id`=`audio/preprocessed/ref16k`
  - format: `target_sample_rate`=`16000`; `input.encoding`=`FLOAT32LE`; `input.bit_depth`=`32`; `input.layout`=`interleaved`; `output.encoding`=`FLOAT32LE`; `output.bit_depth`=`32`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `mic.enabled`=`true`; `ref.enabled`=`false`

### `fa_sample_format`
- 状態: 実装済み。
- 根拠path: `src/processing/format/fa_sample_format/package.xml`, `src/processing/format/fa_sample_format/CMakeLists.txt`, `src/processing/format/fa_sample_format/config/default.yaml`, `src/processing/format/fa_sample_format/launch/fa_sample_format.launch.py`, `src/processing/format/fa_sample_format/src/fa_sample_format_node.cpp`, `src/processing/format/fa_sample_format/test`。
- 実行ファイル/ROS node: exec `fa_sample_format_node`; node `fa_sample_format`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/frame` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/sample_format/mic` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/frame`; `output_topic`=`audio/sample_format/mic`
  - identity/scope: `input_stream_id`=`audio/raw/mic`; `output.stream_id`=`audio/float32/mic`
  - format: `input.encoding`=`PCM16LE`; `input.bit_depth`=`16`; `output.encoding`=`FLOAT32LE`; `output.bit_depth`=`32`; `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### Processing: dynamics

### `fa_agc`
- 状態: 実装済み。
- 根拠path: `src/processing/dynamics/fa_agc/package.xml`, `src/processing/dynamics/fa_agc/CMakeLists.txt`, `src/processing/dynamics/fa_agc/config/default.yaml`, `src/processing/dynamics/fa_agc/launch/fa_agc.launch.py`, `src/processing/dynamics/fa_agc/src/fa_agc_node.cpp`, `src/processing/dynamics/fa_agc/test`。
- 実行ファイル/ROS node: exec `fa_agc_node`; node `fa_agc`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_agc/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_agc/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_agc/input`; `output_topic`=`fa_agc/output`
  - identity/scope: `input_stream_id`=`audio/compressed/mic`; `output.stream_id`=`audio/agc/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`

### `fa_compressor`
- 状態: 実装済み。
- 根拠path: `src/processing/dynamics/fa_compressor/package.xml`, `src/processing/dynamics/fa_compressor/CMakeLists.txt`, `src/processing/dynamics/fa_compressor/config/default.yaml`, `src/processing/dynamics/fa_compressor/launch/fa_compressor.launch.py`, `src/processing/dynamics/fa_compressor/src/fa_compressor_node.cpp`, `src/processing/dynamics/fa_compressor/test`。
- 実行ファイル/ROS node: exec `fa_compressor_node`; node `fa_compressor`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_compressor/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_compressor/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_compressor/input`; `output_topic`=`fa_compressor/output`
  - identity/scope: `input_stream_id`=`audio/normalized/mic`; `output.stream_id`=`audio/compressed/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`

### `fa_expander`
- 状態: 実装済み。
- 根拠path: `src/processing/dynamics/fa_expander/package.xml`, `src/processing/dynamics/fa_expander/CMakeLists.txt`, `src/processing/dynamics/fa_expander/config/default.yaml`, `src/processing/dynamics/fa_expander/launch/fa_expander.launch.py`, `src/processing/dynamics/fa_expander/src/fa_expander_node.cpp`, `src/processing/dynamics/fa_expander/test`。
- 実行ファイル/ROS node: exec `fa_expander_node`; node `fa_expander`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_expander/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_expander/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_expander/input`; `output_topic`=`fa_expander/output`
  - identity/scope: `input_stream_id`=`audio/noise_gated/mic`; `output.stream_id`=`audio/expanded/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`

### `fa_gain`
- 状態: 実装済み。
- 根拠path: `src/processing/dynamics/fa_gain/package.xml`, `src/processing/dynamics/fa_gain/CMakeLists.txt`, `src/processing/dynamics/fa_gain/config/default.yaml`, `src/processing/dynamics/fa_gain/launch/fa_gain.launch.py`, `src/processing/dynamics/fa_gain/src/fa_gain_node.cpp`, `src/processing/dynamics/fa_gain/test`。
- 実行ファイル/ROS node: exec `fa_gain_node`; node `fa_gain`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_gain/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_gain/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_gain/input`; `output_topic`=`fa_gain/output`
  - identity/scope: `input_stream_id`=`audio/resample16k/mic`; `output.stream_id`=`audio/gain/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`

### `fa_limiter`
- 状態: 実装済み。
- 根拠path: `src/processing/dynamics/fa_limiter/package.xml`, `src/processing/dynamics/fa_limiter/CMakeLists.txt`, `src/processing/dynamics/fa_limiter/config/default.yaml`, `src/processing/dynamics/fa_limiter/launch/fa_limiter.launch.py`, `src/processing/dynamics/fa_limiter/src/fa_limiter_node.cpp`, `src/processing/dynamics/fa_limiter/test`。
- 実行ファイル/ROS node: exec `fa_limiter_node`; node `fa_limiter`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_limiter/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_limiter/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_limiter/input`; `output_topic`=`fa_limiter/output`
  - identity/scope: `input_stream_id`=`audio/gain/mic`; `output.stream_id`=`audio/limit/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`

### `fa_noise_gate`
- 状態: 実装済み。
- 根拠path: `src/processing/dynamics/fa_noise_gate/package.xml`, `src/processing/dynamics/fa_noise_gate/CMakeLists.txt`, `src/processing/dynamics/fa_noise_gate/config/default.yaml`, `src/processing/dynamics/fa_noise_gate/launch/fa_noise_gate.launch.py`, `src/processing/dynamics/fa_noise_gate/src/fa_noise_gate_node.cpp`, `src/processing/dynamics/fa_noise_gate/test`。
- 実行ファイル/ROS node: exec `fa_noise_gate_node`; node `fa_noise_gate`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_noise_gate/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_noise_gate/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_noise_gate/input`; `output_topic`=`fa_noise_gate/output`
  - identity/scope: `input_stream_id`=`audio/dc_offset_removed/mic`; `output.stream_id`=`audio/noise_gated/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`

### `fa_normalize`
- 状態: 実装済み。
- 根拠path: `src/processing/dynamics/fa_normalize/package.xml`, `src/processing/dynamics/fa_normalize/CMakeLists.txt`, `src/processing/dynamics/fa_normalize/config/default.yaml`, `src/processing/dynamics/fa_normalize/launch/fa_normalize.launch.py`, `src/processing/dynamics/fa_normalize/src/fa_normalize_node.cpp`, `src/processing/dynamics/fa_normalize/test`。
- 実行ファイル/ROS node: exec `fa_normalize_node`; node `fa_normalize`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_normalize/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_normalize/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_normalize/input`; `output_topic`=`fa_normalize/output`
  - identity/scope: `input_stream_id`=`audio/noise_gated/mic`; `output.stream_id`=`audio/normalized/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`

### Processing: frequency

### `fa_band_pass`
- 状態: 実装済み。
- 根拠path: `src/processing/frequency/fa_band_pass/package.xml`, `src/processing/frequency/fa_band_pass/CMakeLists.txt`, `src/processing/frequency/fa_band_pass/config/default.yaml`, `src/processing/frequency/fa_band_pass/launch/fa_band_pass.launch.py`, `src/processing/frequency/fa_band_pass/src/fa_band_pass_node.cpp`, `src/processing/frequency/fa_band_pass/test`。
- 実行ファイル/ROS node: exec `fa_band_pass_node`; node `fa_band_pass`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_band_pass/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_band_pass/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_band_pass/input`; `output_topic`=`fa_band_pass/output`
  - identity/scope: `input_stream_id`=`audio/sample_format/mic`; `output.stream_id`=`audio/band_pass/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_deesser`
- 状態: 実装済み。
- 根拠path: `src/processing/frequency/fa_deesser/package.xml`, `src/processing/frequency/fa_deesser/CMakeLists.txt`, `src/processing/frequency/fa_deesser/config/default.yaml`, `src/processing/frequency/fa_deesser/launch/fa_deesser.launch.py`, `src/processing/frequency/fa_deesser/src/fa_deesser_node.cpp`, `src/processing/frequency/fa_deesser/test`。
- 実行ファイル/ROS node: exec `fa_deesser_node`; node `fa_deesser`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_deesser/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_deesser/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_deesser/input`; `output_topic`=`fa_deesser/output`
  - identity/scope: `input_stream_id`=`audio/normalized/mic`; `output.stream_id`=`audio/deessed/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`true`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_eq`
- 状態: 実装済み。
- 根拠path: `src/processing/frequency/fa_eq/package.xml`, `src/processing/frequency/fa_eq/CMakeLists.txt`, `src/processing/frequency/fa_eq/config/default.yaml`, `src/processing/frequency/fa_eq/launch/fa_eq.launch.py`, `src/processing/frequency/fa_eq/src/fa_eq_node.cpp`, `src/processing/frequency/fa_eq/test`。
- 実行ファイル/ROS node: exec `fa_eq_node`; node `fa_eq`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_eq/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_eq/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_eq/input`; `output_topic`=`fa_eq/output`
  - identity/scope: `input_stream_id`=`audio/sample_format/mic`; `output.stream_id`=`audio/eq/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_filter`
- 状態: 計画/未実装。
- 根拠path: `src/processing/frequency/fa_filter/README.md`, `src/processing/frequency/fa_filter/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_high_pass`
- 状態: 実装済み。
- 根拠path: `src/processing/frequency/fa_high_pass/package.xml`, `src/processing/frequency/fa_high_pass/CMakeLists.txt`, `src/processing/frequency/fa_high_pass/config/default.yaml`, `src/processing/frequency/fa_high_pass/launch/fa_high_pass.launch.py`, `src/processing/frequency/fa_high_pass/src/fa_high_pass_node.cpp`, `src/processing/frequency/fa_high_pass/test`。
- 実行ファイル/ROS node: exec `fa_high_pass_node`; node `fa_high_pass`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_high_pass/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_high_pass/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_high_pass/input`; `output_topic`=`fa_high_pass/output`
  - identity/scope: `input_stream_id`=`audio/resample16k/mic`; `output.stream_id`=`audio/high_pass/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_low_pass`
- 状態: 実装済み。
- 根拠path: `src/processing/frequency/fa_low_pass/package.xml`, `src/processing/frequency/fa_low_pass/CMakeLists.txt`, `src/processing/frequency/fa_low_pass/config/default.yaml`, `src/processing/frequency/fa_low_pass/launch/fa_low_pass.launch.py`, `src/processing/frequency/fa_low_pass/src/fa_low_pass_node.cpp`, `src/processing/frequency/fa_low_pass/test`。
- 実行ファイル/ROS node: exec `fa_low_pass_node`; node `fa_low_pass`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_low_pass/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_low_pass/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_low_pass/input`; `output_topic`=`fa_low_pass/output`
  - identity/scope: `input_stream_id`=`audio/resample16k/mic`; `output.stream_id`=`audio/low_pass/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_notch`
- 状態: 実装済み。
- 根拠path: `src/processing/frequency/fa_notch/package.xml`, `src/processing/frequency/fa_notch/CMakeLists.txt`, `src/processing/frequency/fa_notch/config/default.yaml`, `src/processing/frequency/fa_notch/launch/fa_notch.launch.py`, `src/processing/frequency/fa_notch/src/fa_notch_node.cpp`, `src/processing/frequency/fa_notch/test`。
- 実行ファイル/ROS node: exec `fa_notch_node`; node `fa_notch`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_notch/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_notch/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_notch/input`; `output_topic`=`fa_notch/output`
  - identity/scope: `input_stream_id`=`audio/high_pass/mic`; `output.stream_id`=`audio/notch/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_spectral_subtraction`
- 状態: 計画/未実装。
- 根拠path: `src/processing/frequency/fa_spectral_subtraction/README.md`, `src/processing/frequency/fa_spectral_subtraction/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_wiener`
- 状態: 計画/未実装。
- 根拠path: `src/processing/frequency/fa_wiener/README.md`, `src/processing/frequency/fa_wiener/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### Processing: correction/noise

### `fa_aec_linear`
- 状態: 実装済み。
- 根拠path: `src/processing/correction/fa_aec_linear/package.xml`, `src/processing/correction/fa_aec_linear/CMakeLists.txt`, `src/processing/correction/fa_aec_linear/config/default.yaml`, `src/processing/correction/fa_aec_linear/launch/fa_aec_linear.launch.py`, `src/processing/correction/fa_aec_linear/src/fa_aec_linear_node.cpp`, `src/processing/correction/fa_aec_linear/test`。
- 実行ファイル/ROS node: exec `fa_aec_linear_node`; node `fa_aec_linear`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `mic_topic` default `audio/resample16k/mic` / `fa_interfaces/msg/AudioFrame`; `ref_topic` default `audio/resample16k/ref` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/aec_linear/frame` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `mic_topic`=`audio/resample16k/mic`; `ref_topic`=`audio/resample16k/ref`; `output_topic`=`audio/aec_linear/frame`
  - identity/scope: `mic_stream_id`=`audio/mic/resample16k`; `ref_stream_id`=`audio/ref/resample16k`; `output.stream_id`=`audio/aec_linear/output`
  - format: `expected_sample_rate`=`16000`; `expected_channels`=`1`; `expected.encoding`=`PCM16LE`; `expected.bit_depth`=`16`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `enabled`=`true`

### `fa_aec_nn`
- 状態: 実装済み。
- 根拠path: `src/processing/correction/fa_aec_nn/package.xml`, `src/processing/correction/fa_aec_nn/CMakeLists.txt`, `src/processing/correction/fa_aec_nn/config/default.yaml`, `src/processing/correction/fa_aec_nn/launch/fa_aec_nn.launch.py`, `src/processing/correction/fa_aec_nn/src/fa_aec_nn_node.cpp`, `src/processing/correction/fa_aec_nn/test`。
- 実行ファイル/ROS node: exec `fa_aec_nn_node`; node `fa_aec_nn`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_aec_nn/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_aec_nn/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_aec_nn/input`; `output_topic`=`fa_aec_nn/output`
  - identity/scope: `input_stream_id`=`audio/aec_linear/frame`; `output.stream_id`=`audio/aec/frame`
  - backend/model/external: `backend.name`=`""`
  - format: `expected_sample_rate`=`16000`; `expected_channels`=`1`; `expected.encoding`=`PCM16LE`; `expected.bit_depth`=`16`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `enabled`=`true`

### `fa_dc_offset_removal`
- 状態: 実装済み。
- 根拠path: `src/processing/correction/fa_dc_offset_removal/package.xml`, `src/processing/correction/fa_dc_offset_removal/CMakeLists.txt`, `src/processing/correction/fa_dc_offset_removal/config/default.yaml`, `src/processing/correction/fa_dc_offset_removal/launch/fa_dc_offset_removal.launch.py`, `src/processing/correction/fa_dc_offset_removal/src/fa_dc_offset_removal_node.cpp`, `src/processing/correction/fa_dc_offset_removal/test`。
- 実行ファイル/ROS node: exec `fa_dc_offset_removal_node`; node `fa_dc_offset_removal`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_dc_offset_removal/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_dc_offset_removal/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_dc_offset_removal/input`; `output_topic`=`fa_dc_offset_removal/output`
  - identity/scope: `input_stream_id`=`audio/sample_format/mic`; `output.stream_id`=`audio/dc_offset_removed/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_debreath`
- 状態: 計画/未実装。
- 根拠path: `src/processing/correction/fa_debreath/README.md`, `src/processing/correction/fa_debreath/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_declick`
- 状態: 実装済み。
- 根拠path: `src/processing/correction/fa_declick/package.xml`, `src/processing/correction/fa_declick/CMakeLists.txt`, `src/processing/correction/fa_declick/config/default.yaml`, `src/processing/correction/fa_declick/launch/fa_declick.launch.py`, `src/processing/correction/fa_declick/src/fa_declick_node.cpp`, `src/processing/correction/fa_declick/test`。
- 実行ファイル/ROS node: exec `fa_declick_node`; node `fa_declick`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_declick/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_declick/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_declick/input`; `output_topic`=`fa_declick/output`
  - identity/scope: `input_stream_id`=`audio/noise_gated/mic`; `output.stream_id`=`audio/declicked/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `window.max_samples`=`1`

### `fa_declip`
- 状態: 計画/未実装。
- 根拠path: `src/processing/correction/fa_declip/README.md`, `src/processing/correction/fa_declip/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_denoise`
- 状態: 実装済み。
- 根拠path: `src/processing/correction/fa_denoise/package.xml`, `src/processing/correction/fa_denoise/CMakeLists.txt`, `src/processing/correction/fa_denoise/config/default.yaml`, `src/processing/correction/fa_denoise/launch/fa_denoise.launch.py`, `src/processing/correction/fa_denoise/src/fa_denoise_node.cpp`, `src/processing/correction/fa_denoise/test`。
- 実行ファイル/ROS node: exec `fa_denoise_node`; node `fa_denoise`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_denoise/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_denoise/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_denoise/input`; `output_topic`=`fa_denoise/output`
  - identity/scope: `input_stream_id`=`audio/resample16k/mic`; `output.stream_id`=`audio/denoised/mic`
  - backend/model/external: `backend.name`=`dtln_onnx`
  - format: `expected_sample_rate`=`16000`; `expected_channels`=`1`; `expected.encoding`=`PCM16LE`; `expected.bit_depth`=`16`; `output.encoding`=`PCM16LE`; `output.bit_depth`=`16`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `enabled`=`true`; `dtln.model_1_path`=`""`; `dtln.model_2_path`=`""`

### `fa_dereverb`
- 状態: 計画/未実装。
- 根拠path: `src/processing/correction/fa_dereverb/README.md`, `src/processing/correction/fa_dereverb/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_hum`
- 状態: 実装済み。
- 根拠path: `src/processing/correction/fa_hum/package.xml`, `src/processing/correction/fa_hum/CMakeLists.txt`, `src/processing/correction/fa_hum/config/default.yaml`, `src/processing/correction/fa_hum/launch/fa_hum.launch.py`, `src/processing/correction/fa_hum/src/fa_hum_node.cpp`, `src/processing/correction/fa_hum/test`。
- 実行ファイル/ROS node: exec `fa_hum_node`; node `fa_hum`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_hum/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_hum/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_hum/input`; `output_topic`=`fa_hum/output`
  - identity/scope: `input_stream_id`=`audio/dc_offset_removed/mic`; `output.stream_id`=`audio/hum_removed/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_wind`
- 状態: 計画/未実装。
- 根拠path: `src/processing/correction/fa_wind/README.md`, `src/processing/correction/fa_wind/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### Processing: temporal

### `fa_crossfade`
- 状態: 実装済み。
- 根拠path: `src/processing/temporal/fa_crossfade/package.xml`, `src/processing/temporal/fa_crossfade/CMakeLists.txt`, `src/processing/temporal/fa_crossfade/config/default.yaml`, `src/processing/temporal/fa_crossfade/launch/fa_crossfade.launch.py`, `src/processing/temporal/fa_crossfade/src/fa_crossfade_node.cpp`, `src/processing/temporal/fa_crossfade/test`。
- 実行ファイル/ROS node: exec `fa_crossfade_node`; node `fa_crossfade`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_a_topic` default `fa_crossfade/input_a` / `fa_interfaces/msg/AudioFrame`; `input_b_topic` default `fa_crossfade/input_b` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_crossfade/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_a_topic`=`fa_crossfade/input_a`; `input_b_topic`=`fa_crossfade/input_b`; `output_topic`=`fa_crossfade/output`
  - identity/scope: `input_a_stream_id`=`audio/segment/a`; `input_b_stream_id`=`audio/segment/b`; `output.stream_id`=`audio/crossfaded/segment`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`

### `fa_delay`
- 状態: 実装済み。
- 根拠path: `src/processing/temporal/fa_delay/package.xml`, `src/processing/temporal/fa_delay/CMakeLists.txt`, `src/processing/temporal/fa_delay/config/default.yaml`, `src/processing/temporal/fa_delay/launch/fa_delay.launch.py`, `src/processing/temporal/fa_delay/src/fa_delay_node.cpp`, `src/processing/temporal/fa_delay/test`。
- 実行ファイル/ROS node: exec `fa_delay_node`; node `fa_delay`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_delay/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_delay/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_delay/input`; `output_topic`=`fa_delay/output`
  - identity/scope: `input_stream_id`=`audio/buffered/mic`; `output.stream_id`=`audio/delayed/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_echo`
- 状態: 実装済み。
- 根拠path: `src/processing/temporal/fa_echo/package.xml`, `src/processing/temporal/fa_echo/CMakeLists.txt`, `src/processing/temporal/fa_echo/config/default.yaml`, `src/processing/temporal/fa_echo/launch/fa_echo.launch.py`, `src/processing/temporal/fa_echo/src/fa_echo_node.cpp`, `src/processing/temporal/fa_echo/test`。
- 実行ファイル/ROS node: exec `fa_echo_node`; node `fa_echo`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_echo/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_echo/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_echo/input`; `output_topic`=`fa_echo/output`
  - identity/scope: `input_stream_id`=`audio/buffered/mic`; `output.stream_id`=`audio/echo/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_fade`
- 状態: 実装済み。
- 根拠path: `src/processing/temporal/fa_fade/package.xml`, `src/processing/temporal/fa_fade/CMakeLists.txt`, `src/processing/temporal/fa_fade/config/default.yaml`, `src/processing/temporal/fa_fade/launch/fa_fade.launch.py`, `src/processing/temporal/fa_fade/src/fa_fade_node.cpp`, `src/processing/temporal/fa_fade/test`。
- 実行ファイル/ROS node: exec `fa_fade_node`; node `fa_fade`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_fade/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_fade/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_fade/input`; `output_topic`=`fa_fade/output`
  - identity/scope: `input_stream_id`=`audio/buffered/mic`; `output.stream_id`=`audio/faded/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `fade.mode`=`fade_in`

### `fa_pitch_shift`
- 状態: 計画/未実装。
- 根拠path: `src/processing/temporal/fa_pitch_shift/README.md`, `src/processing/temporal/fa_pitch_shift/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_reverb`
- 状態: 実装済み。
- 根拠path: `src/processing/temporal/fa_reverb/package.xml`, `src/processing/temporal/fa_reverb/CMakeLists.txt`, `src/processing/temporal/fa_reverb/config/default.yaml`, `src/processing/temporal/fa_reverb/launch/fa_reverb.launch.py`, `src/processing/temporal/fa_reverb/src/fa_reverb_node.cpp`, `src/processing/temporal/fa_reverb/test`。
- 実行ファイル/ROS node: exec `fa_reverb_node`; node `fa_reverb`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_reverb/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_reverb/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_reverb/input`; `output_topic`=`fa_reverb/output`
  - identity/scope: `input_stream_id`=`audio/echo/mic`; `output.stream_id`=`audio/reverb/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_silence_removal`
- 状態: 実装済み。
- 根拠path: `src/processing/temporal/fa_silence_removal/package.xml`, `src/processing/temporal/fa_silence_removal/CMakeLists.txt`, `src/processing/temporal/fa_silence_removal/config/default.yaml`, `src/processing/temporal/fa_silence_removal/launch/fa_silence_removal.launch.py`, `src/processing/temporal/fa_silence_removal/src/fa_silence_removal_node.cpp`, `src/processing/temporal/fa_silence_removal/test`。
- 実行ファイル/ROS node: exec `fa_silence_removal_node`; node `fa_silence_removal`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_silence_removal/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_silence_removal/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_silence_removal/input`; `output_topic`=`fa_silence_removal/output`
  - identity/scope: `input_stream_id`=`audio/buffered/mic`; `output.stream_id`=`audio/silence_removed/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_time_stretch`
- 状態: 計画/未実装。
- 根拠path: `src/processing/temporal/fa_time_stretch/README.md`, `src/processing/temporal/fa_time_stretch/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_trim`
- 状態: 実装済み。
- 根拠path: `src/processing/temporal/fa_trim/package.xml`, `src/processing/temporal/fa_trim/CMakeLists.txt`, `src/processing/temporal/fa_trim/config/default.yaml`, `src/processing/temporal/fa_trim/launch/fa_trim.launch.py`, `src/processing/temporal/fa_trim/src/fa_trim_node.cpp`, `src/processing/temporal/fa_trim/test`。
- 実行ファイル/ROS node: exec `fa_trim_node`; node `fa_trim`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_trim/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_trim/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_trim/input`; `output_topic`=`fa_trim/output`
  - identity/scope: `input_stream_id`=`audio/windowed/mic`; `output.stream_id`=`audio/trimmed/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_window`
- 状態: 実装済み。
- 根拠path: `src/processing/temporal/fa_window/package.xml`, `src/processing/temporal/fa_window/CMakeLists.txt`, `src/processing/temporal/fa_window/config/default.yaml`, `src/processing/temporal/fa_window/launch/fa_window.launch.py`, `src/processing/temporal/fa_window/src/fa_window_node.cpp`, `src/processing/temporal/fa_window/test`。
- 実行ファイル/ROS node: exec `fa_window_node`; node `fa_window`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_window/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_window/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_window/input`; `output_topic`=`fa_window/output`
  - identity/scope: `input_stream_id`=`audio/buffered/mic`; `output.stream_id`=`audio/windowed/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `window.type`=`hann`; `window.expected_frames`=`512`; `window.strict_frame_count`=`true`

### Processing: spatial/channel

### `fa_ambisonics`
- 状態: 計画/未実装。
- 根拠path: `src/processing/spatial/fa_ambisonics/README.md`, `src/processing/spatial/fa_ambisonics/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_beamforming`
- 状態: 実装済み。
- 根拠path: `src/processing/spatial/fa_beamforming/package.xml`, `src/processing/spatial/fa_beamforming/CMakeLists.txt`, `src/processing/spatial/fa_beamforming/config/default.yaml`, `src/processing/spatial/fa_beamforming/launch/fa_beamforming.launch.py`, `src/processing/spatial/fa_beamforming/src/fa_beamforming_node.cpp`, `src/processing/spatial/fa_beamforming/test`。
- 実行ファイル/ROS node: exec `fa_beamforming_node`; node `fa_beamforming`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_beamforming/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_beamforming/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_beamforming/input`; `output_topic`=`fa_beamforming/output`
  - identity/scope: `input_stream_id`=`audio/spatial/mic`; `output.stream_id`=`audio/beamformed/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`4`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`; `output.channels`=`1`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_binaural`
- 状態: 計画/未実装。
- 根拠path: `src/processing/spatial/fa_binaural/README.md`, `src/processing/spatial/fa_binaural/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_downmix`
- 状態: 実装済み。
- 根拠path: `src/processing/spatial/fa_downmix/package.xml`, `src/processing/spatial/fa_downmix/CMakeLists.txt`, `src/processing/spatial/fa_downmix/config/default.yaml`, `src/processing/spatial/fa_downmix/launch/fa_downmix.launch.py`, `src/processing/spatial/fa_downmix/src/fa_downmix_node.cpp`, `src/processing/spatial/fa_downmix/test`。
- 実行ファイル/ROS node: exec `fa_downmix_node`; node `fa_downmix`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_downmix/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_downmix/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_downmix/input`; `output_topic`=`fa_downmix/output`
  - identity/scope: `input_stream_id`=`audio/spatial/mic`; `output.stream_id`=`audio/downmixed/mic`
  - format: `expected.sample_rate`=`16000`; `expected.input_channels`=`4`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`; `output.channels`=`2`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `mode`=`pair_average_to_stereo`

### `fa_pan`
- 状態: 実装済み。
- 根拠path: `src/processing/spatial/fa_pan/package.xml`, `src/processing/spatial/fa_pan/CMakeLists.txt`, `src/processing/spatial/fa_pan/config/default.yaml`, `src/processing/spatial/fa_pan/launch/fa_pan.launch.py`, `src/processing/spatial/fa_pan/src/fa_pan_node.cpp`, `src/processing/spatial/fa_pan/test`。
- 実行ファイル/ROS node: exec `fa_pan_node`; node `fa_pan`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_pan/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_pan/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_pan/input`; `output_topic`=`fa_pan/output`
  - identity/scope: `input_stream_id`=`audio/channel_converted/mic`; `output.stream_id`=`audio/panned/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`2`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_source_separation`
- 状態: 計画/未実装。
- 根拠path: `src/processing/spatial/fa_source_separation/README.md`, `src/processing/spatial/fa_source_separation/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_stereo_widening`
- 状態: 実装済み。
- 根拠path: `src/processing/spatial/fa_stereo_widening/package.xml`, `src/processing/spatial/fa_stereo_widening/CMakeLists.txt`, `src/processing/spatial/fa_stereo_widening/config/default.yaml`, `src/processing/spatial/fa_stereo_widening/launch/fa_stereo_widening.launch.py`, `src/processing/spatial/fa_stereo_widening/src/fa_stereo_widening_node.cpp`, `src/processing/spatial/fa_stereo_widening/test`。
- 実行ファイル/ROS node: exec `fa_stereo_widening_node`; node `fa_stereo_widening`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_stereo_widening/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_stereo_widening/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_stereo_widening/input`; `output_topic`=`fa_stereo_widening/output`
  - identity/scope: `input_stream_id`=`audio/spatial/mic`; `output.stream_id`=`audio/stereo_widened/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`2`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_upmix`
- 状態: 実装済み。
- 根拠path: `src/processing/spatial/fa_upmix/package.xml`, `src/processing/spatial/fa_upmix/CMakeLists.txt`, `src/processing/spatial/fa_upmix/config/default.yaml`, `src/processing/spatial/fa_upmix/launch/fa_upmix.launch.py`, `src/processing/spatial/fa_upmix/src/fa_upmix_node.cpp`, `src/processing/spatial/fa_upmix/test`。
- 実行ファイル/ROS node: exec `fa_upmix_node`; node `fa_upmix`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_upmix/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_upmix/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_upmix/input`; `output_topic`=`fa_upmix/output`
  - identity/scope: `input_stream_id`=`audio/mono/mic/raw`; `output.stream_id`=`audio/upmixed/mic/processed`
  - format: `expected.sample_rate`=`16000`; `expected.input_channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`; `output.channels`=`2`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `mode`=`mono_duplicate`

### Processing: analysis/feature

### `fa_cqt`
- 状態: 実装済み。
- 根拠path: `src/processing/analysis/fa_cqt/package.xml`, `src/processing/analysis/fa_cqt/CMakeLists.txt`, `src/processing/analysis/fa_cqt/config/default.yaml`, `src/processing/analysis/fa_cqt/launch/fa_cqt.launch.py`, `src/processing/analysis/fa_cqt/fa_cqt_py/cqt_node.py`, `src/processing/analysis/fa_cqt/test`。
- 実行ファイル/ROS node: exec `fa_cqt_node`; node `fa_cqt`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/features/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/features/cqt` / `fa_interfaces/msg/CqtFrame`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/features/input`; `output_topic`=`audio/features/cqt`
  - identity/scope: `expected.stream_id`=`audio/preprocessed/mono16k`; `output.stream_id`=`audio/features/cqt/frames`
  - backend/model/external: `backend.name`=`internal_cqt`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`

### `fa_log_mel`
- 状態: 実装済み。
- 根拠path: `src/processing/analysis/fa_log_mel/package.xml`, `src/processing/analysis/fa_log_mel/CMakeLists.txt`, `src/processing/analysis/fa_log_mel/config/default.yaml`, `src/processing/analysis/fa_log_mel/launch/fa_log_mel.launch.py`, `src/processing/analysis/fa_log_mel/fa_log_mel_py/log_mel_node.py`, `src/processing/analysis/fa_log_mel/test`。
- 実行ファイル/ROS node: exec `fa_log_mel_node`; node `fa_log_mel`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/features/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/features/log_mel` / `fa_interfaces/msg/LogMelFrame`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/features/input`; `output_topic`=`audio/features/log_mel`
  - identity/scope: `expected.stream_id`=`audio/preprocessed/mono16k`; `output.stream_id`=`audio/features/log_mel/frames`
  - backend/model/external: `backend.name`=`internal_log_mel`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`

### `fa_loudness`
- 状態: 実装済み。
- 根拠path: `src/processing/analysis/fa_loudness/package.xml`, `src/processing/analysis/fa_loudness/CMakeLists.txt`, `src/processing/analysis/fa_loudness/config/default.yaml`, `src/processing/analysis/fa_loudness/launch/fa_loudness.launch.py`, `src/processing/analysis/fa_loudness/fa_loudness_py/loudness_node.py`, `src/processing/analysis/fa_loudness/test`。
- 実行ファイル/ROS node: exec `fa_loudness_node`; node `fa_loudness`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/features/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/features/loudness` / `fa_interfaces/msg/LoudnessFrame`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/features/input`; `output_topic`=`audio/features/loudness`
  - identity/scope: `expected.stream_id`=`audio/preprocessed/mono16k`; `output.stream_id`=`audio/features/loudness/frames`
  - backend/model/external: `backend.name`=`internal_frame_meter`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`

### `fa_mfcc`
- 状態: 実装済み。
- 根拠path: `src/processing/analysis/fa_mfcc/package.xml`, `src/processing/analysis/fa_mfcc/CMakeLists.txt`, `src/processing/analysis/fa_mfcc/config/default.yaml`, `src/processing/analysis/fa_mfcc/launch/fa_mfcc.launch.py`, `src/processing/analysis/fa_mfcc/fa_mfcc_py/mfcc_node.py`, `src/processing/analysis/fa_mfcc/test`。
- 実行ファイル/ROS node: exec `fa_mfcc_node`; node `fa_mfcc`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/features/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/features/mfcc` / `fa_interfaces/msg/MfccFrame`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/features/input`; `output_topic`=`audio/features/mfcc`
  - identity/scope: `expected.stream_id`=`audio/preprocessed/mono16k`; `output.stream_id`=`audio/features/mfcc/frames`
  - backend/model/external: `backend.name`=`internal_mfcc`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`

### `fa_onset`
- 状態: 実装済み。
- 根拠path: `src/processing/analysis/fa_onset/package.xml`, `src/processing/analysis/fa_onset/CMakeLists.txt`, `src/processing/analysis/fa_onset/config/default.yaml`, `src/processing/analysis/fa_onset/launch/fa_onset.launch.py`, `src/processing/analysis/fa_onset/fa_onset_py/onset_node.py`, `src/processing/analysis/fa_onset/test`。
- 実行ファイル/ROS node: exec `fa_onset_node`; node `fa_onset`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/features/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/features/onset` / `fa_interfaces/msg/OnsetFrame`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/features/input`; `output_topic`=`audio/features/onset`
  - identity/scope: `expected.stream_id`=`audio/preprocessed/mono16k`; `output.stream_id`=`audio/features/onset/frames`
  - backend/model/external: `backend.name`=`internal_spectral_flux`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`

### `fa_pitch`
- 状態: 実装済み。
- 根拠path: `src/processing/analysis/fa_pitch/package.xml`, `src/processing/analysis/fa_pitch/CMakeLists.txt`, `src/processing/analysis/fa_pitch/config/default.yaml`, `src/processing/analysis/fa_pitch/launch/fa_pitch.launch.py`, `src/processing/analysis/fa_pitch/fa_pitch_py/pitch_node.py`, `src/processing/analysis/fa_pitch/test`。
- 実行ファイル/ROS node: exec `fa_pitch_node`; node `fa_pitch`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/features/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/features/pitch` / `fa_interfaces/msg/PitchFrame`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/features/input`; `output_topic`=`audio/features/pitch`
  - identity/scope: `expected.stream_id`=`audio/preprocessed/mono16k`; `output.stream_id`=`audio/features/pitch/frames`
  - backend/model/external: `backend.name`=`internal_autocorrelation`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`

### `fa_stft`
- 状態: 実装済み。
- 根拠path: `src/processing/analysis/fa_stft/package.xml`, `src/processing/analysis/fa_stft/CMakeLists.txt`, `src/processing/analysis/fa_stft/config/default.yaml`, `src/processing/analysis/fa_stft/launch/fa_stft.launch.py`, `src/processing/analysis/fa_stft/fa_stft_py/stft_node.py`, `src/processing/analysis/fa_stft/test`。
- 実行ファイル/ROS node: exec `fa_stft_node`; node `fa_stft`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/features/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/features/stft` / `fa_interfaces/msg/StftFrame`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/features/input`; `output_topic`=`audio/features/stft`
  - identity/scope: `expected.stream_id`=`audio/preprocessed/mono16k`; `output.stream_id`=`audio/features/stft/frames`
  - backend/model/external: `backend.name`=`internal_stft`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`

### `fa_tempo`
- 状態: 実装済み。
- 根拠path: `src/processing/analysis/fa_tempo/package.xml`, `src/processing/analysis/fa_tempo/CMakeLists.txt`, `src/processing/analysis/fa_tempo/config/default.yaml`, `src/processing/analysis/fa_tempo/launch/fa_tempo.launch.py`, `src/processing/analysis/fa_tempo/fa_tempo_py/tempo_node.py`, `src/processing/analysis/fa_tempo/test`。
- 実行ファイル/ROS node: exec `fa_tempo_node`; node `fa_tempo`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/features/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/features/tempo` / `fa_interfaces/msg/TempoFrame`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/features/input`; `output_topic`=`audio/features/tempo`
  - identity/scope: `expected.stream_id`=`audio/preprocessed/mono16k`; `output.stream_id`=`audio/features/tempo/frames`
  - backend/model/external: `backend.name`=`internal_onset_autocorrelation`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`

### Processing: generation

### `fa_music_source_separation`
- 状態: 計画/未実装。
- 根拠path: `src/processing/generation/fa_music_source_separation/README.md`, `src/processing/generation/fa_music_source_separation/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_neural_codec`
- 状態: 計画/未実装。
- 根拠path: `src/processing/generation/fa_neural_codec/README.md`, `src/processing/generation/fa_neural_codec/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_neural_vocoder`
- 状態: 計画/未実装。
- 根拠path: `src/processing/generation/fa_neural_vocoder/README.md`, `src/processing/generation/fa_neural_vocoder/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_speech_enhancement`
- 状態: 計画/未実装。
- 根拠path: `src/processing/generation/fa_speech_enhancement/README.md`, `src/processing/generation/fa_speech_enhancement/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_speech_separation`
- 状態: 計画/未実装。
- 根拠path: `src/processing/generation/fa_speech_separation/README.md`, `src/processing/generation/fa_speech_separation/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_speech_translation`
- 状態: 計画/未実装。
- 根拠path: `src/processing/generation/fa_speech_translation/README.md`, `src/processing/generation/fa_speech_translation/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_super_resolution`
- 状態: 計画/未実装。
- 根拠path: `src/processing/generation/fa_super_resolution/README.md`, `src/processing/generation/fa_super_resolution/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### `fa_tts`
- 状態: 実装済み。
- 根拠path: `src/processing/generation/fa_tts/package.xml`, `src/processing/generation/fa_tts/CMakeLists.txt`, `src/processing/generation/fa_tts/config/default.yaml`, `src/processing/generation/fa_tts/launch/fa_tts.launch.py`, `src/processing/generation/fa_tts/fa_tts_py/tts_node.py`, `src/processing/generation/fa_tts/test`。
- 実行ファイル/ROS node: exec `fa_tts_node`; node `fa_tts`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: なし。
- Publishers: `output_topic` default `audio/tts/frame` / `fa_interfaces/msg/AudioFrame`
- Services: `speak` / `fa_interfaces/srv/Speak`
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `output_topic`=`audio/tts/frame`
  - identity/scope: `output.source_id`=`fa_tts`; `output.stream_id`=`tts_synthesis`
  - backend/model/external: `backend.name`=`pyopenjtalk`; `backend.openjtalk_dict_dir`=`""`; `default_voice`=`""`; `cache_dir`=`~/.cache/fluent_audio/tts`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`true`

### `fa_voice_conversion`
- 状態: 計画/未実装。
- 根拠path: `src/processing/generation/fa_voice_conversion/README.md`, `src/processing/generation/fa_voice_conversion/test`。
- External runtime API: 外部runtime APIは未実装。package.xml / 実行ノード / launch / config は未作成。

### Processing: routing/mixing

### `fa_bus_router`
- 状態: 実装済み。
- 根拠path: `src/processing/routing/fa_bus_router/package.xml`, `src/processing/routing/fa_bus_router/CMakeLists.txt`, `src/processing/routing/fa_bus_router/config/default.yaml`, `src/processing/routing/fa_bus_router/launch/fa_bus_router.launch.py`, `src/processing/routing/fa_bus_router/src/fa_bus_router_node.cpp`, `src/processing/routing/fa_bus_router/test`。
- 実行ファイル/ROS node: exec `fa_bus_router_node`; node `fa_bus_router`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_bus_router/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topics` default `["fa_bus_router/output"]` / `fa_interfaces/msg/AudioFrame[]`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_bus_router/input`; `output_topics`=`["fa_bus_router/output"]`
  - identity/scope: `input_stream_id`=`audio/routing/input`; `output.stream_ids`=`["audio/routing/output"]`
  - format: `expected.sample_rate`=`48000`; `expected.channels`=`1`; `expected.encoding`=`PCM16LE`; `expected.bit_depth`=`16`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`true`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_ducking`
- 状態: 実装済み。
- 根拠path: `src/processing/routing/fa_ducking/package.xml`, `src/processing/routing/fa_ducking/CMakeLists.txt`, `src/processing/routing/fa_ducking/config/default.yaml`, `src/processing/routing/fa_ducking/launch/fa_ducking.launch.py`, `src/processing/routing/fa_ducking/src/fa_ducking_node.cpp`, `src/processing/routing/fa_ducking/test`。
- 実行ファイル/ROS node: exec `fa_ducking_node`; node `fa_ducking`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `program_topic` default `fa_ducking/program` / `fa_interfaces/msg/AudioFrame`; `sidechain_topic` default `fa_ducking/sidechain` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_ducking/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `program_topic`=`fa_ducking/program`; `sidechain_topic`=`fa_ducking/sidechain`; `output_topic`=`fa_ducking/output`
  - identity/scope: `program_stream_id`=`audio/program/frame`; `sidechain_stream_id`=`audio/sidechain/frame`; `output.stream_id`=`audio/ducked/frame`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`

### `fa_loopback`
- 状態: 実装済み。
- 根拠path: `src/processing/routing/fa_loopback/package.xml`, `src/processing/routing/fa_loopback/CMakeLists.txt`, `src/processing/routing/fa_loopback/config/default.yaml`, `src/processing/routing/fa_loopback/launch/fa_loopback.launch.py`, `src/processing/routing/fa_loopback/src/fa_loopback_node.cpp`, `src/processing/routing/fa_loopback/test`。
- 実行ファイル/ROS node: exec `fa_loopback_node`; node `fa_loopback`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_loopback/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_loopback/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_loopback/input`; `output_topic`=`fa_loopback/output`
  - identity/scope: `input_stream_id`=`audio/output/frame`; `output.stream_id`=`audio/loopback/frame`
  - format: `expected.sample_rate`=`48000`; `expected.channels`=`1`; `expected.encoding`=`PCM16LE`; `expected.bit_depth`=`16`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`true`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`

### `fa_mix`
- 状態: 実装済み。
- 根拠path: `src/processing/routing/fa_mix/package.xml`, `src/processing/routing/fa_mix/CMakeLists.txt`, `src/processing/routing/fa_mix/config/default.yaml`, `src/processing/routing/fa_mix/launch/fa_mix.launch.py`, `src/processing/routing/fa_mix/src/fa_mix_node.cpp`, `src/processing/routing/fa_mix/test`。
- 実行ファイル/ROS node: exec `fa_mix_node`; node `fa_mix`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topics` default `["fa_mix/tts"]` / `fa_interfaces/msg/AudioFrame[]`
- Publishers: `output_topic` default `fa_mix/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topics`=`["fa_mix/tts"]`; `output_topic`=`fa_mix/output`
  - identity/scope: `input_stream_ids`=`["audio/tts/frame"]`; `output.stream_id`=`audio/mix/output`
  - format: `expected.sample_rate`=`48000`; `expected.channels`=`1`; `expected.bit_depth`=`16`; `expected.encoding`=`PCM16LE`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`true`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`
  - other public: `input_gains_db`=`[0.0]`

### `fa_monitor_mix`
- 状態: 実装済み。
- 根拠path: `src/processing/routing/fa_monitor_mix/package.xml`, `src/processing/routing/fa_monitor_mix/CMakeLists.txt`, `src/processing/routing/fa_monitor_mix/config/default.yaml`, `src/processing/routing/fa_monitor_mix/launch/fa_monitor_mix.launch.py`, `src/processing/routing/fa_monitor_mix/src/fa_monitor_mix_node.cpp`, `src/processing/routing/fa_monitor_mix/test`。
- 実行ファイル/ROS node: exec `fa_monitor_mix_node`; node `fa_monitor_mix`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topics` default `["audio/program/frame", "audio/tts/frame"]` / `fa_interfaces/msg/AudioFrame[]`
- Publishers: `output_topic` default `audio/monitor/frame` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topics`=`["audio/program/frame", "audio/tts/frame"]`; `output_topic`=`audio/monitor/frame`
  - identity/scope: `input_stream_ids`=`["audio/program_bus", "audio/tts_bus"]`; `output.stream_id`=`audio/monitor_bus`; `output.source_id`=`monitor_mix`
  - format: `expected.sample_rate`=`48000`; `expected.channels`=`2`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `input_gains_db`=`[0.0]`

### `fa_patchbay`
- 状態: 実装済み。
- 根拠path: `src/processing/routing/fa_patchbay/package.xml`, `src/processing/routing/fa_patchbay/CMakeLists.txt`, `src/processing/routing/fa_patchbay/config/default.yaml`, `src/processing/routing/fa_patchbay/launch/fa_patchbay.launch.py`, `src/processing/routing/fa_patchbay/src/fa_patchbay_node.cpp`, `src/processing/routing/fa_patchbay/test`。
- 実行ファイル/ROS node: exec `fa_patchbay_node`; node `fa_patchbay`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topics` default `["fa_patchbay/input"]` / `fa_interfaces/msg/AudioFrame[]`
- Publishers: `output_topics` default `["fa_patchbay/output"]` / `fa_interfaces/msg/AudioFrame[]`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topics`=`["fa_patchbay/input"]`; `output_topics`=`["fa_patchbay/output"]`
  - identity/scope: `input_stream_ids`=`["audio/patchbay/input"]`; `output_stream_ids`=`["audio/patchbay/output"]`
  - format: `expected.sample_rate`=`48000`; `expected.channels`=`1`; `expected.encoding`=`PCM16LE`; `expected.bit_depth`=`16`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`true`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_sidechain`
- 状態: 実装済み。
- 根拠path: `src/processing/routing/fa_sidechain/package.xml`, `src/processing/routing/fa_sidechain/CMakeLists.txt`, `src/processing/routing/fa_sidechain/config/default.yaml`, `src/processing/routing/fa_sidechain/launch/fa_sidechain.launch.py`, `src/processing/routing/fa_sidechain/src/fa_sidechain_node.cpp`, `src/processing/routing/fa_sidechain/test`。
- 実行ファイル/ROS node: exec `fa_sidechain_node`; node `fa_sidechain`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `sidechain_topic` default `fa_sidechain/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `control_topic` default `fa_sidechain/control` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `sidechain_topic`=`fa_sidechain/input`; `control_topic`=`fa_sidechain/control`
  - identity/scope: `sidechain_stream_id`=`audio/sidechain/frame`; `control.stream_id`=`audio/sidechain/control`
  - format: `control.sample_rate`=`1000`; `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`false`
  - other public: `detector.active_gain_db`=`-12.0`; `detector.inactive_gain_db`=`0.0`

## Streaming/Synchronization

### `fa_audio_window`
- 状態: 実装済み。
- 根拠path: `src/streaming/fa_audio_window/package.xml`, `src/streaming/fa_audio_window/CMakeLists.txt`, `src/streaming/fa_audio_window/config/default.yaml`, `src/streaming/fa_audio_window/launch/fa_audio_window.launch.py`, `src/streaming/fa_audio_window/src/fa_audio_window_node.cpp`, `src/streaming/fa_audio_window/test`。
- 実行ファイル/ROS node: exec `fa_audio_window_node`; node `fa_audio_window`。
- Launch: `launch/fa_audio_window.launch.py` は package share の `config/default.yaml` を使い、node name `fa_audio_window` で起動する。
- Subscriptions: `input_topic` default `fa_audio_window/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: なし。
- Services: `service_name` default `export_audio_window` / `fa_interfaces/srv/ExportAudioWindow`; `archive_service_name` default `archive_audio_window` / `fa_interfaces/srv/ArchiveAudioWindow`
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_audio_window/input`; `service_name`=`export_audio_window`; `archive_service_name`=`archive_audio_window`
  - identity/scope: `input.source_id`=`mic`; `input.stream_id`=`audio/mic`; `window.id`=`fa_audio_window`; `window.epoch`=`1`; `audio.default_scope`=`mic`; `audio.supported_scopes`=`["mic"]`
  - format: `expected.encoding`=`PCM16LE`; `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.bit_depth`=`16`; `expected.layout`=`interleaved`; `export.codec`=`pcm_s16le`; `export.container`=`wav`; `export.payload_format`=`audio/wav`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`
  - other public: `window.retention_seconds`=`1800`; `export.output_directory`=`/tmp/fa_audio_window`; `archive.store.backend`=`local_file`; `archive.store.directory`=`""`; `archive.store.uri_prefix`=`""`; `archive.store.metadata_uri_prefix`=`""`

### `fa_chunk_overlap`
- 状態: 実装済み。
- 根拠path: `src/streaming/fa_chunk_overlap/package.xml`, `src/streaming/fa_chunk_overlap/CMakeLists.txt`, `src/streaming/fa_chunk_overlap/config/default.yaml`, `src/streaming/fa_chunk_overlap/launch/fa_chunk_overlap.launch.py`, `src/streaming/fa_chunk_overlap/src/fa_chunk_overlap_node.cpp`, `src/streaming/fa_chunk_overlap/test`。
- 実行ファイル/ROS node: exec `fa_chunk_overlap_node`; node `fa_chunk_overlap`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_chunk_overlap/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_chunk_overlap/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_chunk_overlap/input`; `output_topic`=`fa_chunk_overlap/output`
  - identity/scope: `input_stream_id`=`audio/float32le/mic`; `output.stream_id`=`audio/chunked_overlap/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`true`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `window.frame_samples`=`512`; `window.hop_samples`=`256`

### `fa_clock_drift`
- 状態: 実装済み。
- 根拠path: `src/streaming/fa_clock_drift/package.xml`, `src/streaming/fa_clock_drift/CMakeLists.txt`, `src/streaming/fa_clock_drift/config/default.yaml`, `src/streaming/fa_clock_drift/launch/fa_clock_drift.launch.py`, `src/streaming/fa_clock_drift/src/fa_clock_drift_node.cpp`, `src/streaming/fa_clock_drift/test`。
- 実行ファイル/ROS node: exec `fa_clock_drift_node`; node `fa_clock_drift`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_clock_drift/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_clock_drift/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_clock_drift/input`; `output_topic`=`fa_clock_drift/output`
  - identity/scope: `input_stream_id`=`audio/sample_format/mic`; `output.stream_id`=`audio/clock_drift_corrected/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_frame_buffer`
- 状態: 実装済み。
- 根拠path: `src/streaming/fa_frame_buffer/package.xml`, `src/streaming/fa_frame_buffer/CMakeLists.txt`, `src/streaming/fa_frame_buffer/config/default.yaml`, `src/streaming/fa_frame_buffer/launch/fa_frame_buffer.launch.py`, `src/streaming/fa_frame_buffer/src/fa_frame_buffer_node.cpp`, `src/streaming/fa_frame_buffer/test`。
- 実行ファイル/ROS node: exec `fa_frame_buffer_node`; node `fa_frame_buffer`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_frame_buffer/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_frame_buffer/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_frame_buffer/input`; `output_topic`=`fa_frame_buffer/output`
  - identity/scope: `input_stream_id`=`audio/noise_gated/mic`; `output.stream_id`=`audio/buffered/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_jitter_buffer`
- 状態: 実装済み。
- 根拠path: `src/streaming/fa_jitter_buffer/package.xml`, `src/streaming/fa_jitter_buffer/CMakeLists.txt`, `src/streaming/fa_jitter_buffer/config/default.yaml`, `src/streaming/fa_jitter_buffer/launch/fa_jitter_buffer.launch.py`, `src/streaming/fa_jitter_buffer/src/fa_jitter_buffer_node.cpp`, `src/streaming/fa_jitter_buffer/test`。
- 実行ファイル/ROS node: exec `fa_jitter_buffer_node`; node `fa_jitter_buffer_node`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_jitter_buffer/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_jitter_buffer/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_jitter_buffer/input`; `output_topic`=`fa_jitter_buffer/output`
  - identity/scope: `input_stream_id`=`audio/network/mic`; `output.stream_id`=`audio/jitter_buffered/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_latency_compensation`
- 状態: 実装済み。
- 根拠path: `src/streaming/fa_latency_compensation/package.xml`, `src/streaming/fa_latency_compensation/CMakeLists.txt`, `src/streaming/fa_latency_compensation/config/default.yaml`, `src/streaming/fa_latency_compensation/launch/fa_latency_compensation.launch.py`, `src/streaming/fa_latency_compensation/src/fa_latency_compensation_node.cpp`, `src/streaming/fa_latency_compensation/test`。
- 実行ファイル/ROS node: exec `fa_latency_compensation_node`; node `fa_latency_compensation`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `audio/frame` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `audio/latency_compensated/frame` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`audio/frame`; `output_topic`=`audio/latency_compensated/frame`
  - identity/scope: `input_stream_id`=`audio/preprocessed/mono16k`; `output.stream_id`=`audio/latency_compensated/mono16k`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_overlap_add`
- 状態: 実装済み。
- 根拠path: `src/streaming/fa_overlap_add/package.xml`, `src/streaming/fa_overlap_add/CMakeLists.txt`, `src/streaming/fa_overlap_add/config/default.yaml`, `src/streaming/fa_overlap_add/launch/fa_overlap_add.launch.py`, `src/streaming/fa_overlap_add/src/fa_overlap_add_node.cpp`, `src/streaming/fa_overlap_add/test`。
- 実行ファイル/ROS node: exec `fa_overlap_add_node`; node `fa_overlap_add_node`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_overlap_add/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_overlap_add/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_overlap_add/input`; `output_topic`=`fa_overlap_add/output`
  - identity/scope: `input_stream_id`=`audio/chunked_overlap/mic`; `output.stream_id`=`audio/overlap_added/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`true`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`
  - other public: `window.frame_samples`=`512`; `window.hop_samples`=`256`; `window.type`=`hann`

### `fa_packet_loss_concealment`
- 状態: 実装済み。
- 根拠path: `src/streaming/fa_packet_loss_concealment/package.xml`, `src/streaming/fa_packet_loss_concealment/CMakeLists.txt`, `src/streaming/fa_packet_loss_concealment/config/default.yaml`, `src/streaming/fa_packet_loss_concealment/launch/fa_packet_loss_concealment.launch.py`, `src/streaming/fa_packet_loss_concealment/src/fa_packet_loss_concealment_node.cpp`, `src/streaming/fa_packet_loss_concealment/test`。
- 実行ファイル/ROS node: exec `fa_packet_loss_concealment_node`; node `fa_packet_loss_concealment_node`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_packet_loss_concealment/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_packet_loss_concealment/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_packet_loss_concealment/input`; `output_topic`=`fa_packet_loss_concealment/output`
  - identity/scope: `input_stream_id`=`audio/stream/input`; `output.stream_id`=`audio/stream/plc`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

### `fa_time_alignment`
- 状態: 実装済み。
- 根拠path: `src/streaming/fa_time_alignment/package.xml`, `src/streaming/fa_time_alignment/CMakeLists.txt`, `src/streaming/fa_time_alignment/config/default.yaml`, `src/streaming/fa_time_alignment/launch/fa_time_alignment.launch.py`, `src/streaming/fa_time_alignment/src/fa_time_alignment_node.cpp`, `src/streaming/fa_time_alignment/test`。
- 実行ファイル/ROS node: exec `fa_time_alignment_node`; node `fa_time_alignment`。
- Launch: package launch は通常 `node_name` と `config_file` を引数に取り、`parameters=[config_file]` で node を起動する。
- Subscriptions: `input_topic` default `fa_time_alignment/input` / `fa_interfaces/msg/AudioFrame`
- Publishers: `output_topic` default `fa_time_alignment/output` / `fa_interfaces/msg/AudioFrame`; `diagnostics` / `diagnostic_msgs/msg/DiagnosticArray`
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - topic/service: `input_topic`=`fa_time_alignment/input`; `output_topic`=`fa_time_alignment/output`
  - identity/scope: `input_stream_id`=`audio/frame_buffer/mic`; `output.stream_id`=`audio/time_aligned/mic`
  - format: `expected.sample_rate`=`16000`; `expected.channels`=`1`; `expected.encoding`=`FLOAT32LE`; `expected.bit_depth`=`32`; `expected.layout`=`interleaved`
  - QoS: `qos.depth`=`10`; `qos.reliable`=`false`; `diagnostics.qos.depth`=`10`; `diagnostics.qos.reliable`=`true`

## System/Profiles

### `fluent_audio_system`
- 状態: 基盤。
- 根拠path: `src/system/fluent_audio_system/package.xml`, `src/system/fluent_audio_system/setup.py`, `src/system/fluent_audio_system/launch/fluent_audio_system.launch.py`, `src/system/fluent_audio_system/launch/run.py`, `src/system/fluent_audio_system/test`。
- 実行ファイル/ROS node: exec `list_required_packages`。
- Launch: `launch/run.py` が public wrapper、`launch/fluent_audio_system.launch.py` が system YAML expansion 本体。
- Subscriptions: なし。
- Publishers: なし。
- Services: なし。
- Clients / MCP tools: なし。
- Public config parameters:
  - launch引数: `config`, `fa_in_enabled`, `fa_out_enabled`, `fa_in_source_id`, `fa_out_sink_id`。
  - system YAML schema: `system.default_start_delay`, `system.inter_group_delay`, `groups[].id`, `groups[].enable`, `groups[].nodes[]`。
  - node schema: `id`, `enable`, `package`, `exec`, `node_name`, `namespace`, `output`, `params_file`, `parameters`, `remappings`, `env`。
  - profile/config: `config/fluent_audio_system.sample.yaml`, `config/profiles/so101*.yaml`。
