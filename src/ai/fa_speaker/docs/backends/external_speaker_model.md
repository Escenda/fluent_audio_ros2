# external_speaker_model backend

`external_speaker_model` は speaker embedding / identification / verification model を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `embedding.dimension`
- `enrolled_speaker_index`

## Forbidden

- ROS2 topic/message dependency inside backend
- unknown speaker fallback
- missing model fallback
- ASR / diarization timeline の混入

