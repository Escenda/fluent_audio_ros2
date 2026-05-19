# external_speaker_model backend

`external_speaker_model` は speaker embedding / identification / verification model を external worker / process / container として扱う backend contract である。
`fa_speaker` は package 化前の roadmap note であり、この backend contract は package 化時に fake backend behavioral tests で固定する。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- supported sample rates
- supported window
- execution mode: batch speaker embedding または streaming speaker verification
- output schema: speaker embedding、speaker id、verification score、model id、source identity、stream identity
- `embedding.dimension`
- `enrolled_speaker_index`

## Supported AudioFrame Input

`external_speaker_model` が受け取れる入力は、config と model capability が明示する `FLOAT32LE`、`bit_depth=32`、mono、interleaved、configured sample rate、configured window の `AudioFrame` に限定する。
`PCM16`、`PCM32`、stereo、planar、sample-rate mismatch、window mismatch、non-finite sample、range violation は変換せず fail closed または frame reject とする。

## Model Capability

- model id
- model path、file artifact が必要な場合のみ
- provider / worker command または endpoint
- supported sample rates
- supported window
- batch / streaming mode
- output schema
- embedding dimension
- enrolled speaker index

## Forbidden

- ROS2 topic/message dependency inside backend
- unknown speaker fallback
- missing model fallback
- ASR / diarization timeline の混入
