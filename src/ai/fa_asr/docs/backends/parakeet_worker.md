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
- `backend.args`: `{audio}`、`{model}`、`{sample_rate}` を含む
- `backend.health_args`: startup health check。`{model}` を含む

## Boundary

`fa_asr` は一時 raw float32le `.f32` file を作り、worker command に path と sample rate を渡します。worker は transcript を stdout または `backend.output_text_path` に出力します。PCM16 / WAV 変換は worker 側または明示された前段 node 側の責務です。

## Failure Conditions

- command path missing
- model id missing
- health args missing / malformed
- health check non-zero exit / timeout
- worker non-zero exit
- timeout
- empty transcript

失敗時に local command や Whisper へ切り替えません。
