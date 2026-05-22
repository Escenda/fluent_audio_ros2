# src/apps

Application-layer packages live under explicit responsibility directories:

- `voice_command/`: command routing and mode control.
- `agent_tools/`: adapter packages that expose FluentAudio capabilities to external agent/tool runtimes.
- `dialogue/`: dialogue orchestration that joins wake word, turn detection, TTS, and external reasoning services.
- `safety/`: safety policy for audio-driven commands.

Device I/O and audio processing do not live here.
Agent tool adapters do not own audio DSP, model runtimes, timeline buffers, or safety decisions.

## Package Status

Only directories with `package.xml` are ROS 2 packages.

| Directory | Status |
| --- | --- |
| `voice_command/fa_voice_command_router/` | ROS 2 package |
| `agent_tools/fa_audio_mcp/` | ROS 2 package |
| `dialogue/fa_dialogue/` | ROS 2 package |
| `safety/fa_safety_policy/` | roadmap placeholder; not a ROS 2 package |
