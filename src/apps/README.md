# src/apps

Application-layer packages live under explicit responsibility directories:

- `voice_command/`: command routing and mode control.
- `dialogue/`: dialogue orchestration that joins wake word, ASR, turn detection, TTS, and external reasoning services.
- `safety/`: safety policy for audio-driven commands.

Device I/O and audio processing do not live here.
