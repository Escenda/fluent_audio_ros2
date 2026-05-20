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
- `backend.health_args`: 任意。指定時は startup health check として `{model}` を含む command を実行する
- `backend.result_format`: `plain_text` または `segments_json_v1`

## Input

- mono float samples
- sample rate

## Command Contract

backend は一時 raw float32le `.f32` file path と sample rate を command に渡し、stdout または output file から `backend.result_format` に従う結果を読みます。default config は `backend.result_format` を空にし、output contract を暗黙選択しません。

`plain_text` では worker output を transcript text として扱い、`fa_asr` が selected ASR request samples 全体を覆う 1 segment にします。`segments_json_v1` では worker output を strict JSON として扱い、top-level は `result_format` と `segments` だけを許可します。各 segment は integer `start_sample` / `end_sample`、非空 `text`、任意の非空 `speaker_label` だけを持ち、offset は selected ASR request samples からの相対 sample index です。

invalid JSON、schema 不一致、範囲不正、overlap / unsorted order、空 text、空 `speaker_label` は fail closed です。`fa_asr` は worker output を推測、補正、型 coercion、resample、downmix、PCM/int16 変換して成功扱いにしません。

## Failure Conditions

- command path missing
- health check non-zero exit / timeout
- non-zero exit
- timeout
- invalid result format output
- empty transcript / empty segment text
