# parakeet_worker Backend

## Backend Name

`parakeet_worker`

## Runtime

Parakeet 系 ASR を外部 worker / process / container として呼び出します。`fa_asr` は NeMo / PyTorch / Parakeet の Python package を import しません。
`ParakeetWorkerAsrBackend` は専用 class であり、`LocalCommandAsrBackend` の alias ではありません。subprocess 実行のみ内部 helper を共有します。

## Required Config

- `backend.command`: worker CLI
- `backend.model`: worker に渡す model id
- `backend.language`
- `backend.args`: `{audio}` と `{model}` を含む

## Boundary

`fa_asr` は一時 WAV file を作り、worker command に path を渡します。worker は transcript を stdout または `backend.output_text_path` に出力します。

## Failure Conditions

- command path missing
- model id missing
- worker non-zero exit
- timeout
- empty transcript

失敗時に local command や Whisper へ切り替えません。
