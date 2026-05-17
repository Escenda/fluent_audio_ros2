# openai_transcriptions Backend

## Backend Name

`openai_transcriptions`

## Runtime

OpenAI Transcriptions API を外部 worker / process / container として呼び出す backend slot です。`fa_asr` は OpenAI SDK、API key 読み取り、network retry を持ちません。
`OpenAiTranscriptionsAsrBackend` は専用 class であり、`LocalCommandAsrBackend` の alias ではありません。subprocess 実行のみ内部 helper を共有します。

## Required Config

- `backend.command`: external OpenAI Transcriptions worker CLI
- `backend.model`: worker に渡す transcription model id
- `backend.language`
- `backend.args`: `{audio}` と `{model}` を含む

## Boundary

OpenAI API key、network endpoint、request format、response parsing は worker 側の責務です。`fa_asr` は validated WAV path と model id を渡し、transcript text を受け取ります。

## Failure Conditions

- command path missing
- model id missing
- worker non-zero exit
- timeout
- empty transcript

OpenAI Transcriptions backend が失敗しても local backend へ fallback しません。
