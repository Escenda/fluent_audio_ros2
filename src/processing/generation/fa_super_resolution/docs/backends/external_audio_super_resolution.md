# external_audio_super_resolution backend

`external_audio_super_resolution` は audio bandwidth extension model を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `target.sample_rate`
- `target.bandwidth_hz`

## Forbidden

- ROS2 topic/message dependency inside backend
- simple resample fallback
- EQ fallback
- missing model fallback

