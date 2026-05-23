# fa_turn_detector

`fa_turn_detector` estimates whether the user has finished their turn through an external Smart Turn v3 ONNX worker. It publishes a turn-end candidate only after `fa_dialogue` sends a `TurnEndRequest` for the active turn.

## 入出力

- Sub: configured `audio_topic` (`fa_interfaces/msg/AudioFrame`)
- Sub: `conversation/turn_context` (`fa_interfaces/msg/TurnContext`)
- Sub: `voice/turn_end_request` (`fa_interfaces/msg/TurnEndRequest`)
- Pub: `voice/turn_end` (`fa_interfaces/msg/TurnEnd`)

## QoS

QoS は edge ごとに明示します。depth は正の整数、reliable は bool として扱い、node 内で topic 名から推測しません。

```yaml
audio.qos.depth: 10
audio.qos.reliable: false
turn_context.qos.depth: 10
turn_context.qos.reliable: true
turn_end_request.qos.depth: 10
turn_end_request.qos.reliable: true
output.qos.depth: 10
output.qos.reliable: true
```

## Runtime

ONNX Runtime は ROS2 node process では import しません。`backend.command` で指定した external worker 側の Python / venv / container に明示的に provision します。`backend.name: smart_turn_onnx` では ONNX model path、execution provider、worker command、inference args、health-check args が必須です。空 model path から package share の model は推測しません。
