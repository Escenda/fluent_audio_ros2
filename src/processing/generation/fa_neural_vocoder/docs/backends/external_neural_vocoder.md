# external_neural_vocoder backend

`external_neural_vocoder` は neural vocoder engine を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `feature_schema`
- `payload_encoding`

## Forbidden

- ROS2 topic/message dependency inside backend
- feature schema auto-reshape
- default speaker conditioning fallback
- missing model fallback

