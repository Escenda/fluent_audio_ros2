# external_speech_translation backend

`external_speech_translation` は speech-to-speech translation engine を external worker / process / container として扱う backend contract である。

## Required Config

- `backend.name`
- `backend.command` または `backend.endpoint`
- `model_path` または `model_id`
- `source_language`
- `target_language`

## Forbidden

- ROS2 topic/message dependency inside backend
- ASR-only fallback
- TTS-only fallback
- missing model fallback

