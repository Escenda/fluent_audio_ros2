# whisper_cpp Backend

## Backend Name

`whisper_cpp`

## Runtime

`whisper.cpp` の CLI を subprocess として実行します。`fa_asr` は whisper の Python package を import しません。

## Required Config

- `backend.command`: `whisper-cli` などの実行ファイル
- `backend.model_path`: ggml model file path
- `backend.language`
- `backend.args`: `{audio}` と `{model}` を含む

## Failure Conditions

- command path missing
- model file missing
- `{audio}` or `{model}` placeholder missing
- non-zero exit
- timeout
- empty transcript

失敗時に別 ASR backend へ切り替えません。
