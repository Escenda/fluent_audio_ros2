# sherpa_onnx_kws Backend

## Backend Name

`sherpa_onnx_kws`

## Runtime

C++ / sherpa-onnx C API。

## Input

- mono float samples
- sample rate
- VAD probability

## Output

- detected
- keyword
- score (`1.0` for binary sherpa-onnx keyword result)

## Required Files

- encoder model
- decoder model
- joiner model
- tokens file
- keywords file

Missing file は起動失敗です。

## Required Runtime Parameters

- `backend.execution_provider`: `cpu`, `cuda`, or `coreml`

空値または未対応 provider は sherpa-onnx C API に渡す前に起動失敗です。
