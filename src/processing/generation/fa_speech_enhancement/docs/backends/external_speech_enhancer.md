# external_speech_enhancer backend

`external_speech_enhancer` は neural speech enhancement engine を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `payload_encoding`
- `expected.sample_rate`

## Forbidden

- ROS2 topic/message dependency inside backend
- enhancement failure の input pass-through
- missing model fallback
- AEC / dereverb / VAD の混入

