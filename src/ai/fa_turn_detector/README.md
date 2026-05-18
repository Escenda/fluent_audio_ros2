# fa_turn_detector

`fa_turn_detector` estimates whether the user has finished their turn through an external Smart Turn v3 ONNX worker.

## 入出力

- Sub: configured `audio_topic` (`fa_interfaces/msg/AudioFrame`)
- Sub: `voice/vad_state` (`fa_interfaces/msg/VadState`)
- Sub: `conversation/turn_context` (`fa_interfaces/msg/TurnContext`)
- Pub: `voice/turn_end` (`fa_interfaces/msg/TurnEnd`)

`TurnContext.active=true` の間だけ音声をバッファします。`AudioFrame.source_id` と `VadState.source_id` は `expected_source_id`、`AudioFrame.stream_id` と `VadState.stream_id` は `expected_stream_id` に一致する必要があります。`audio_topic` は ROS transport の接続点であり、frame identity ではありません。モデルファイルが無い場合は起動時に失敗します。

## Runtime

ONNX Runtime は ROS2 node process では import しません。`backend.command` で指定した external worker 側の Python / venv / container に明示的に provision します。`backend.name: smart_turn_onnx` では ONNX model path、execution provider、worker command、inference args、health-check args が必須です。空 model path から package share の model は推測しません。
