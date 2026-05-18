# external_sed backend

`external_sed` は sound event detection model を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `label_set`
- `payload_encoding`

## Forbidden

- ROS2 topic/message dependency inside backend
- unknown label fallback
- missing model fallback
- VAD / KWS / ASR の混入

