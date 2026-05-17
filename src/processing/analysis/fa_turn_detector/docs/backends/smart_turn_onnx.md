# smart_turn_onnx Backend

## Backend Name

`smart_turn_onnx`

## Runtime

Python / ONNX Runtime。

## Input

- mono float samples
- sample rate
- explicit ONNX Runtime execution provider

## Output

- turn-end probability

## Failure Conditions

- model path missing
- execution provider missing / unavailable
- invalid ONNX model
- unsupported feature shape

Missing model は fallback せず起動失敗です。
