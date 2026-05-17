# openai_realtime Backend

## Backend Name

`openai_realtime`

## Runtime

OpenAI Realtime ASR を外部 worker / process / container として呼び出す backend slot です。`fa_asr` は OpenAI SDK、WebSocket client、network client を持ちません。`backend.openai_realtime.api_key_env` で指定された環境変数が空でないことだけを起動時に検証します。
`OpenAiRealtimeAsrBackend` は専用 class であり、`LocalCommandAsrBackend` の alias ではありません。subprocess 実行のみ内部 helper を共有します。

## Required Config

- `backend.command`: external OpenAI Realtime worker CLI
- `backend.model`: worker に渡す model id
- `backend.openai_realtime.api_key_env`: OpenAI API key を保持する環境変数名
- `backend.language`
- `backend.args`: `{audio}`、`{model}`、`{sample_rate}` を含む

## Boundary

OpenAI API key の値、network endpoint、Realtime session、WebSocket/WebRTC 接続は worker 側の責務です。`fa_asr` は key 値を読まず、環境変数名と存在だけを検証します。その後、validated raw float32le `.f32` path、model id、sample rate を渡し、transcript text を受け取ります。

## Failure Conditions

- command path missing
- model id missing
- `backend.openai_realtime.api_key_env` missing
- referenced API key environment variable missing or empty
- worker non-zero exit
- timeout
- empty transcript

OpenAI backend が失敗しても local backend へ fallback しません。
