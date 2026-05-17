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
- `vad_threshold`: finite value in `(0.0, 1.0]`

空値または未対応 provider は sherpa-onnx C API に渡す前に起動失敗です。
`vad_threshold` が 0 以下、1 超過、または NaN の場合も backend 初期化失敗です。VAD gate を閾値 0 で無効化する経路は提供しません。
`vad_prob` が `vad_threshold` 未満の場合、backend は sherpa-onnx stream に波形を渡す前に stream を reset して `nullopt` を返します。
