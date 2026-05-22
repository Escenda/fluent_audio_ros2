# src/apps/dialogue

detection, TTS, and external reasoning services, but they do not own model
runtime backends or low-level audio processing.

`fa_dialogue` is currently the minimal turn context publisher. It joins
`conversation/turn_context`; reasoning, TTS, safety policy, and robot command
proposal remain separate application slices.
