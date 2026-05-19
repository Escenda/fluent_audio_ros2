# external_sed backend

`external_sed` は sound event detection model を external worker / process / container として扱う backend contract である。
`fa_sed` は package 化前の roadmap note であり、この backend contract は package 化時に fake backend behavioral tests で固定する。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- supported sample rates
- supported window
- execution mode: batch detection または streaming detection
- output schema: event label、confidence、time range、model id、source identity、stream identity
- `label_set`
- `payload_encoding`

## Supported AudioFrame Input

`external_sed` が受け取れる入力は、config と model capability が明示する `FLOAT32LE`、`bit_depth=32`、mono、interleaved、configured sample rate、configured window の `AudioFrame` に限定する。
`PCM16`、`PCM32`、stereo、planar、sample-rate mismatch、window mismatch、non-finite sample、range violation は変換せず fail closed または frame reject とする。

## Model Capability

- model id
- model path、file artifact が必要な場合のみ
- provider / worker command または endpoint
- supported sample rates
- supported window
- batch / streaming mode
- output schema
- label set

## Forbidden

- ROS2 topic/message dependency inside backend
- unknown label fallback
- missing model fallback
- VAD / KWS / ASR の混入
