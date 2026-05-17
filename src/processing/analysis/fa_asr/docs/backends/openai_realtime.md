# openai_realtime Backend

## Backend Name

`openai_realtime`

## Runtime

OpenAI Realtime ASR を外部 worker / process / container として呼び出す backend slot です。`fa_asr` は OpenAI SDK、WebSocket client、API key 読み取りを持ちません。

## Required Config

- `backend.command`: external OpenAI Realtime worker CLI
- `backend.model`: worker に渡す model id
- `backend.language`
- `backend.args`: `{audio}` と `{model}` を含む

## Boundary

OpenAI API key、network endpoint、Realtime session、WebSocket/WebRTC 接続は worker 側の責務です。`fa_asr` は validated WAV path と model id を渡し、transcript text を受け取ります。

## Failure Conditions

- command path missing
- model id missing
- worker non-zero exit
- timeout
- empty transcript

OpenAI backend が失敗しても local backend へ fallback しません。
