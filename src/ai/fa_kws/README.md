# fa_kws

`fa_kws` は sherpa-onnx のローカル KWS モデルを使うウェイクワード/キーワードスポッティングノードです。

## 入出力

- Sub: configured `audio_topic` (`fa_interfaces/msg/AudioFrame`)
- Sub: optional `control.<id>.topic` (`fa_interfaces/msg/VoiceActivity`)
- Pub: configured `output_topic` (`fa_interfaces/msg/WakeWordResult`)

## モデル

`config/default.yaml` の `model.encoder` / `model.decoder` / `model.joiner` / `model.tokens` / `kws.keywords_file` は必須です。空または存在しないパスを指定した場合、ノードは起動時に失敗します。`backend.execution_provider` も必須で、空または未対応 provider は external worker に渡す前に失敗します。

`expected_source_id` と `expected_stream_id` も必須です。受信した `AudioFrame.source_id` と `AudioFrame.stream_id` は設定値と一致する必要があります。`audio_topic` は ROS transport の接続点であり、frame identity ではありません。別 source / stream の audio frame は backend に渡さず reject します。

audio / output topic QoS は `audio.qos.*`、`output.qos.*` で明示します。depth が 0 以下の場合は起動失敗し、node code 内の hidden depth / reliability へ切り替えません。

`control.inputs` に `fa_interfaces/msg/VoiceActivity` gate を指定すると、KWS backend には VAD が許可した audio chunk だけを渡します。`stream_gate.*` は pre-roll、hangover、短い VAD gap の merge を扱う node/controller 層の責務で、backend と external worker には VAD topic や turn lifecycle を渡しません。

`fa_kws_node` は sherpa-onnx C API を直接 link しません。`backend.command` / `backend.args` / `backend.stream_args` / `backend.health_args` で外部 worker を明示し、その worker が sherpa-onnx runtime、Python / C++ runtime、venv、container、GPU provider を所有します。`backend.stream_args` がある場合は worker を常駐起動し、VAD によって開いた区間だけ `PUSH` します。worker 欠落、health check failure、timeout、invalid stdout は起動または推論時に fail closed します。

同梱の `scripts/sherpa_onnx_kws_worker` は Python `sherpa_onnx` runtime 向け reference worker entrypoint です。
実体は `fa_kws_py/backends/sherpa_onnx_kws_worker.py` に分離し、script は thin entrypoint として扱います。

通常の workspace build では `fa_kws_node` と `fa_kws_wav_tool` を build します。native sherpa-onnx link mode、unavailable backend、dummy backend、別 backend への暗黙切り替えはありません。
