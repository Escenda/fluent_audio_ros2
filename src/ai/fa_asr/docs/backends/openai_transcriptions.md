# openai_transcriptions Backend

## Backend Name

`openai_transcriptions`

## Runtime

OpenAI Transcriptions API を外部 worker / process / container として呼び出す backend slot です。`fa_asr` は OpenAI SDK、network retry、OpenAI API client を持ちません。`backend.openai_transcriptions.api_key_env` で指定された環境変数が空でないことを起動時に検証し、worker process には canonical な `OPENAI_API_KEY` 環境変数として渡します。
`OpenAiTranscriptionsAsrBackend` は専用 class であり、`LocalCommandAsrBackend` の alias ではありません。subprocess 実行のみ内部 helper を共有します。

## Required Config

- `backend.command`: external OpenAI Transcriptions worker CLI
- `backend.model`: worker に渡す transcription model id
- `backend.openai_transcriptions.api_key_env`: OpenAI API key を保持する環境変数名
- `backend.language`
- `backend.args`: `{audio}`、`{model}`、`{sample_rate}` を含む
- `backend.health_args`: startup health check。`{model}` を含む

## Boundary

OpenAI API 呼び出し、network endpoint、request format、response parsing は worker 側の責務です。`fa_asr` は key 値を API client として使わず、configured env の値を `OPENAI_API_KEY` として worker process environment に注入します。key 値は argv、stdout、stderr、ROS topic、ログへ渡しません。その後、validated raw float32le `.f32` path、model id、sample rate を argv で渡し、transcript text を受け取ります。

## Failure Conditions

- command path missing
- model id missing
- `backend.openai_transcriptions.api_key_env` missing
- referenced API key environment variable missing or empty
- health args missing / malformed
- health check non-zero exit / timeout
- worker non-zero exit
- timeout
- empty transcript

OpenAI Transcriptions backend が失敗しても local backend へ fallback しません。
