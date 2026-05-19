# sherpa_onnx_kws Backend

## Backend Name

`sherpa_onnx_kws`

## Runtime

External worker / process。

`fa_kws_node` は sherpa-onnx C API を link しません。`backend.command` で指定した worker process に canonical float32le audio file と model/config arguments を渡し、stdout の最終非空行から detection result を読みます。sherpa-onnx runtime、Python / C++ runtime、venv、container、GPU provider は worker 側の責務です。

同梱の `scripts/sherpa_onnx_kws_worker` は Python `sherpa_onnx` runtime 向けの reference worker です。ROS2 node process からは直接 import されず、別 venv / container に置いた同じ CLI contract の worker に差し替えられます。

native sherpa-onnx link mode、unavailable backend、dummy backend、別 backend への fallback は提供しません。

## Input

- mono float samples
- sample rate
- VAD probability

## Output

- `NO_DETECTION`
- `DETECTED<TAB>keyword<TAB>score<TAB>start_time_sec`

invalid stdout、non-zero exit、timeout は fail closed です。空 transcript や壊れた数値を no detection として扱いません。

## Required Files

- encoder model
- decoder model
- joiner model
- tokens file
- keywords file

Missing file は起動失敗です。
directory や読み取り不能 file を model / tokens / keywords として扱うことも禁止し、起動失敗にします。

## Required Runtime Parameters

- `backend.name`: `sherpa_onnx_kws`
- `backend.execution_provider`: `cpu`, `cuda`, or `coreml`
- `vad.probability_gate`: finite value in `(0.0, 1.0]`
- `target_sample_rate`: positive sample rate
- `model.encoder`: readable regular file
- `model.decoder`: readable regular file
- `model.joiner`: readable regular file
- `model.tokens`: readable regular file
- `kws.keywords_file`: readable regular file
- `model.num_threads`: positive thread count
- `kws.max_active_paths`: positive path count
- `kws.num_trailing_blanks`: zero or positive blank count
- `kws.keywords_score`: finite positive score
- `kws.keywords_threshold`: finite positive threshold
- `backend.command`: executable worker command
- `backend.args`: inference command args。`{audio}`, `{encoder}`, `{decoder}`, `{joiner}`, `{tokens}`, `{keywords}`, `{provider}`, `{sample_rate}`, `{num_threads}`, `{max_active_paths}`, `{num_trailing_blanks}`, `{keywords_score}`, `{keywords_threshold}` を含む
- `backend.health_args`: startup health-check args。`{encoder}`, `{decoder}`, `{joiner}`, `{tokens}`, `{keywords}`, `{provider}`, `{sample_rate}`, `{num_threads}`, `{max_active_paths}`, `{num_trailing_blanks}`, `{keywords_score}`, `{keywords_threshold}` を含み、`{audio}` を含まない
- `backend.timeout_sec`: finite positive timeout
- `backend.workspace_dir`: audio payload workspace
- `backend.cleanup_audio_files`: boolean
- `output.qos.depth`: positive depth
- `output.qos.reliable`: boolean

空値または未対応 provider は worker に渡す前に起動失敗です。
`vad.probability_gate` が 0 以下、1 超過、または NaN の場合も backend 初期化失敗です。VAD gate を閾値 0 で無効化する経路は提供しません。
`VadState.probability` が `vad.probability_gate` 未満の場合、backend は worker を呼ばずに `nullopt` を返します。

`kws_wav_tool` もこの backend config を直接使うため、CLI で `--command`, `--workspace_dir`, `--timeout_sec`, `--sample_rate`, `--num_threads`, `--max_active_paths`, `--num_trailing_blanks`, `--keywords_score`, `--threshold` を明示します。未指定値を struct の default や未初期化 memory に依存して補完しません。
