# external_separator backend

`external_separator` は neural / DSP source separation engine を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `target_sources`
- `payload_encoding`

## Forbidden

- ROS2 topic/message dependency inside backend
- target source list の自動推測
- missing model fallback
- speaker embedding / diarization の混入

