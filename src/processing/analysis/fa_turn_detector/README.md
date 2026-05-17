# fa_turn_detector

`fa_turn_detector` runs a local Smart Turn v3 ONNX model to estimate whether the user has finished their turn.

## 入出力

- Sub: `audio/frame` (`fa_interfaces/msg/AudioFrame`)
- Sub: `voice/vad_state` (`fa_interfaces/msg/VadState`)
- Sub: `conversation/turn_context` (`fa_interfaces/msg/TurnContext`)
- Pub: `voice/turn_end` (`fa_interfaces/msg/TurnEnd`)

`TurnContext.active=true` の間だけ音声をバッファします。モデルファイルが無い場合は起動時に失敗します。
