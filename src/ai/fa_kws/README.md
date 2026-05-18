# fa_kws

`fa_kws` は sherpa-onnx のローカル KWS モデルを使うウェイクワード/キーワードスポッティングノードです。

## 入出力

- Sub: configured `audio_topic` (`fa_interfaces/msg/AudioFrame`)
- Sub: configured `vad_topic` (`fa_interfaces/msg/VadState`)
- Pub: configured `output_topic` (`fa_interfaces/msg/WakeWordResult`)

## モデル

`config/default.yaml` の `model.encoder` / `model.decoder` / `model.joiner` / `model.tokens` / `kws.keywords_file` は必須です。空または存在しないパスを指定した場合、ノードは起動時に失敗します。`backend.execution_provider` も必須で、空または未対応 provider は sherpa-onnx C API に渡す前に失敗します。KWS は VAD state を必須入力として扱い、未受信または `vad.max_age_ms` を超えた stale state では audio frame を処理しません。

`expected_source_id` と `expected_stream_id` も必須です。受信した `AudioFrame.source_id` と `VadState.source_id` は `expected_source_id` と一致し、`AudioFrame.stream_id` と `VadState.stream_id` は `expected_stream_id` と一致する必要があります。`audio_topic` は ROS transport の接続点であり、frame identity ではありません。別 source / stream の audio frame または VAD state は backend に渡さず reject します。

audio / VAD / result topic QoS は `audio.qos.*`、`vad.qos.*`、`result.qos.*` で明示します。depth が 0 以下の場合は起動失敗し、node code 内の hidden depth / reliability へ切り替えません。

ビルド時には sherpa-onnx C API が必要です。標準パスに無い場合は `SHERPA_ONNX_PREFIX` 環境変数または CMake cache で install prefix を指定してください。

Contract test だけを実行する環境では、`-DFA_KWS_SHERPA_ONNX=OFF` を明示できます。この場合は `fa_kws_node` / `fa_kws_wav_tool` runtime target 自体を build しません。runtime build では既定の `-DFA_KWS_SHERPA_ONNX=ON` を使い、sherpa-onnx C API が無い場合は configure で失敗します。別 backend や dummy backend へ暗黙に切り替えることはありません。
