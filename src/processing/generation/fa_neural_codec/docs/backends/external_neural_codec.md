# external_neural_codec backend

`external_neural_codec` は neural codec model を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `mode`
- `payload_encoding`

## Forbidden

- ROS2 topic/message dependency inside backend
- transport codec fallback
- missing model fallback
- TTS / vocoder behavior の混入

