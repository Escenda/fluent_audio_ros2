# src/apps

Application-layer packages live under explicit responsibility directories:

- `voice_command/`: command routing and mode control.
- `dialogue/`: dialogue orchestration that joins wake word, ASR, turn detection, TTS, and external reasoning services.
- `safety/`: safety policy for audio-driven commands.

Device I/O and audio processing do not live here.

## Package Status

Only directories with `package.xml` are ROS 2 packages.

| Directory | Status |
| --- | --- |
| `voice_command/fa_voice_command_router/` | ROS 2 package |
| `dialogue/fa_dialogue/` | roadmap placeholder; not a ROS 2 package |
| `safety/fa_safety_policy/` | roadmap placeholder; not a ROS 2 package |
