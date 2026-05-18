# external_music_separator backend

`external_music_separator` は neural music separation engine を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `target_stems`
- `payload_encoding`

## Forbidden

- ROS2 topic/message dependency inside backend
- target stem list の自動推測
- missing model fallback
- routing mixer の混入

