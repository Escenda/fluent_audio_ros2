# parakeet_worker Backend

## Position

`parakeet_worker` is a legacy optional external worker backend.
It is not the current standard ASR backend, not a default path, and not a fallback for `parakeet_multilingual_buffered`.
The current standard multilingual Parakeet path keeps the runner inside `parakeet_multilingual_buffered`.

## Backend Name

`parakeet_worker`

## Runtime

Parakeet 系 ASR を外部 worker / process / container として呼び出します。`fa_asr` は NeMo / PyTorch / Parakeet の Python package を import しません。
`ParakeetWorkerAsrBackend` は専用 class であり、`LocalCommandAsrBackend` の alias ではありません。subprocess 実行のみ内部 helper を共有します。
この backend は non-streaming command backend です。
`nemo_rnnt_streaming` のような streaming session、encoder cache、previous hypothesis、decoder state、JSONL `start/audio/drain/finish` protocol は持ちません。

## Required Config

- `backend.command`: worker CLI
- `backend.model`: worker に渡す model id
- `backend.language`
- `backend.args`: `{audio}`、`{model}`、`{sample_rate}` を含む
- `backend.health_args`: startup health check。`{model}` を含む
- `backend.result_format`: `plain_text` または `segments_json_v1`

## Boundary

`fa_asr` は一時 raw float32le `.f32` file を作り、worker command に path と sample rate を渡します。worker は `backend.result_format` に従う結果を stdout または `backend.output_text_path` に出力します。default config は `backend.result_format` を空にし、output contract を暗黙選択しません。

`plain_text` は transcript text、`segments_json_v1` は top-level が `result_format` と `segments` だけの strict JSON です。`segments_json_v1` の segment offset は selected ASR request samples からの相対 sample index であり、integer `start_sample` / `end_sample`、非空 `text`、任意の非空 `speaker_label` だけを許可します。invalid JSON/schema/range、overlap、空 text は fail closed です。`fa_asr` は PCM16 / WAV 変換、resample、downmix、型 coercion、推測補正を行いません。

この backend が扱わない境界は次です。

- streaming session / cache-aware streaming state
- encoder cache / previous hypothesis / previous decoder output
- NeMo / PyTorch / Parakeet package import inside `fa_asr`
- NIM / Riva server endpoint
- gRPC / WebSocket ASR serving endpoint
- NGC artifact download / `download-version`
- local `.nemo` restore / finite attention context validation

Parakeet model family、NIM / Riva support matrix、NGC artifact metadata は、この command backend の成功証明ではありません。
`parakeet_worker` の成立条件は command health check、raw float32le input file contract、`backend.result_format` に従う output、timeout / non-zero exit / invalid output の fail-closed handling です。

## Failure Conditions

- command path missing
- model id missing
- health args missing / malformed
- health check non-zero exit / timeout
- worker non-zero exit
- timeout
- invalid result format output
- empty transcript / empty segment text

失敗時に local command や Whisper へ切り替えません。
