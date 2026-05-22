# whisper.cpp Backend

## Position

`whisper.cpp` is a legacy optional external worker backend.
It is not the current standard ASR backend, not a default path, not the validation reference, and not a fallback for `parakeet_multilingual_buffered`.

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
- `backend.health_args`: package 単体では任意。profile template で ASR を enable 可能にする場合は明示する。指定時は startup health check として `{model}` を含む command を実行する
- `backend.result_format`: `plain_text` または `segments_json_v1`

## Output Contract

worker は `backend.result_format` に従う結果を stdout または `backend.output_text_path` に出力します。default config は `backend.result_format` を空にし、output contract を暗黙選択しません。`plain_text` は transcript text、`segments_json_v1` は top-level が `result_format` と `segments` だけの strict JSON です。`segments_json_v1` の segment offset は selected ASR request samples からの相対 sample index であり、integer `start_sample` / `end_sample`、非空 `text`、任意の非空 `speaker_label` だけを許可します。invalid JSON/schema/range、overlap、空 text は fail closed です。`fa_asr` は backend output を推測、補正、型 coercion、resample、downmix、PCM/int16 変換して成功扱いにしません。

## Failure Conditions

- command path missing
- model file missing
- health check non-zero exit / timeout
- `{audio}` / `{model}` / `{sample_rate}` placeholder missing
- non-zero exit
- timeout
- invalid result format output
- empty transcript / empty segment text

失敗時に別 ASR backend へ切り替えません。
