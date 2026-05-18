# external_voice_conversion backend

`external_voice_conversion` は voice conversion model を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `target_voice`
- `payload_encoding`

## Forbidden

- ROS2 topic/message dependency inside backend
- default target voice fallback
- missing model fallback
- TTS / ASR の混入

