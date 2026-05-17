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
- score

## Required Files

- encoder model
- decoder model
- joiner model
- tokens file
- keywords file

Missing file は起動失敗です。
