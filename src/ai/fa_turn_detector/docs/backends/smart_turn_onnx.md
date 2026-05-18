# smart_turn_onnx Backend

## Backend Name

`smart_turn_onnx`

## Runtime

External Python / ONNX Runtime worker。

`fa_turn_detector` の ROS2 node process は ONNX Runtime を import しません。`backend.command` で指定した worker process が model load、provider selection、mel feature extraction、ONNX inference を担当します。

## Input

- mono float32 samples serialized as `.npy`
- sample rate
- explicit ONNX Runtime execution provider。adapter 境界で `CPUExecutionProvider`、`CUDAExecutionProvider`、`TensorrtExecutionProvider` だけを受け付ける
- model path passed as `{model}`
- audio payload path passed as `{audio}`
- execution provider passed as `{provider}`

## Output

- turn-end probability

## Failure Conditions

- model path missing
- execution provider missing / unavailable in worker
- unsupported execution provider name
- command missing / not executable
- `backend.args` missing `{audio}`, `{model}`, `{provider}`
- `backend.health_args` missing `{model}`, `{provider}`
- unknown / malformed placeholder in `backend.args` or `backend.health_args`
- invalid ONNX model
- incompatible model input/output contract
- worker timeout / non-zero exit
- worker stdout が probability float ではない
- probability が finite ではない、または `[0.0, 1.0]` の範囲外

Missing model は fallback せず起動失敗です。
startup health check は worker を起動し、ONNX Runtime provider と model IO contract を検証します。health check が失敗した場合も fallback せず起動失敗です。
