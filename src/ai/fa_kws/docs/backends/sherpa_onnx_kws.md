# sherpa_onnx_kws Backend

## Backend Name

`sherpa_onnx_kws`

## Runtime

C++ / sherpa-onnx C API。

CMake の `FA_KWS_SHERPA_ONNX` は `ON` / `OFF` を取る。

- `OFF`: default。`fa_kws_node` / `fa_kws_wav_tool` は build し、`backend.name=sherpa_onnx_kws` 選択時に fail closed する。
- `ON`: sherpa-onnx C API を必須にする。C API が見つからない場合は configure で失敗する。

sherpa-onnx が無い環境で別 model、別 backend、dummy backend へ暗黙に切り替えることはしない。
sherpa-onnx support が無い build で `sherpa_onnx_kws` が選択された場合は、unavailable backend が明示的に起動失敗します。

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
directory や読み取り不能 file を model / tokens / keywords として扱うことも禁止し、起動失敗にします。

## Required Runtime Parameters

- `backend.execution_provider`: `cpu`, `cuda`, or `coreml`
- `vad_threshold`: finite value in `(0.0, 1.0]`
- `target_sample_rate`: positive sample rate
- `model_num_threads`: positive thread count
- `max_active_paths`: positive path count
- `num_trailing_blanks`: zero or positive blank count
- `keywords_score`: finite positive score
- `keywords_threshold`: finite positive threshold

空値または未対応 provider は sherpa-onnx C API に渡す前に起動失敗です。
`vad_threshold` が 0 以下、1 超過、または NaN の場合も backend 初期化失敗です。VAD gate を閾値 0 で無効化する経路は提供しません。
`vad_prob` が `vad_threshold` 未満の場合、backend は sherpa-onnx stream に波形を渡す前に stream を reset して `nullopt` を返します。

`kws_wav_tool` もこの backend config を直接使うため、CLI で `--sample_rate`, `--num_threads`, `--max_active_paths`, `--num_trailing_blanks`, `--keywords_score`, `--threshold` を明示します。未指定値を struct の default や未初期化 memory に依存して補完しません。
