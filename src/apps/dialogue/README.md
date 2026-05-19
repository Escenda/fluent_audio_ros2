# src/apps/dialogue

Dialogue orchestration packages live here. They may join wake word, ASR, turn
detection, TTS, and external reasoning services, but they do not own model
runtime backends or low-level audio processing.

`fa_dialogue` is currently the minimal turn context publisher. It joins
`voice/wake_word`, `voice/asr/result`, and `voice/turn_end` into
`conversation/turn_context`; reasoning, TTS, safety policy, and robot command
proposal remain separate application slices.
