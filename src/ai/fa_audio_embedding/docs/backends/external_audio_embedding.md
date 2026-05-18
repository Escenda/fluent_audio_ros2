# external_audio_embedding backend

`external_audio_embedding` は audio embedding model を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `embedding.dimension`
- `payload_encoding`

## Forbidden

- ROS2 topic/message dependency inside backend
- zero vector fallback
- stale embedding reuse
- missing model fallback

