# local_command Backend

## Backend Name

`local_command`

## Runtime

外部 command を subprocess として実行します。ROS2 node は engine の Python package を import しません。
`LocalCommandAsrBackend` は `local_command` 専用 class であり、他 backend の alias には使いません。

## Required Config

- `backend.command`: 実行ファイル
- `backend.model_path`: model file path
- `backend.language`
- `backend.args`: `{audio}`、`{model}`、`{sample_rate}` を含む

## Input

- mono float samples
- sample rate

## Command Contract

backend は一時 raw float32le `.f32` file path と sample rate を command に渡し、stdout または output file から transcript を読む構造です。PCM16 / WAV 変換は行いません。

## Failure Conditions

- command path missing
- non-zero exit
- timeout
- empty transcript when empty result is disallowed
