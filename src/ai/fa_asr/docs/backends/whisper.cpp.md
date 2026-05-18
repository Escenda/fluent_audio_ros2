# whisper.cpp Backend

## Backend Name

`whisper.cpp`

## Runtime

`whisper.cpp` を呼び出す external worker / wrapper を subprocess として実行します。`fa_asr` は whisper の Python package を import せず、`whisper-cli` が期待する WAV/PCM16 への暗黙変換も行いません。
`WhisperCppAsrBackend` は専用 class であり、`LocalCommandAsrBackend` の alias ではありません。subprocess 実行のみ内部 helper を共有します。

## Required Config

- `backend.command`: raw float32le payload と sample rate を受け取れる `whisper.cpp` worker / wrapper
- `backend.model_path`: ggml model file path
- `backend.language`
- `backend.args`: `{audio}`、`{model}`、`{sample_rate}` を含む
- `backend.health_args`: 任意。指定時は startup health check として `{model}` を含む command を実行する

## Failure Conditions

- command path missing
- model file missing
- health check non-zero exit / timeout
- `{audio}` / `{model}` / `{sample_rate}` placeholder missing
- non-zero exit
- timeout
- empty transcript

失敗時に別 ASR backend へ切り替えません。
