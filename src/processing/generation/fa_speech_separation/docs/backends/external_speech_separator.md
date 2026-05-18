# external_speech_separator backend

`external_speech_separator` は neural speech separation engine を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `target_sources`
- `payload_encoding`

## Forbidden

- ROS2 topic/message dependency inside backend
- source count の自動推測
- missing model fallback
- diarization / ASR の混入

