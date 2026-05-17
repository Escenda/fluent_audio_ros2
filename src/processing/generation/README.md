# Generation Processing

This category contains data-plane nodes that produce or transform audio into a
new audio representation.

Examples:

- text-to-speech
- voice conversion
- speech enhancement
- speech separation
- speech-to-speech translation
- neural codec
- neural vocoder
- audio super-resolution

Generation backends may use external workers, separate Python environments, or
cloud APIs, but the selected backend and required artifacts must be explicit.
The backend code stays ROS-free under each package's `backends/` directory.

This category is not the home for recognition, classification, dialogue policy,
LLM/MCP/VLM orchestration, or application behavior. AI recognition nodes belong
under `src/ai`; application orchestration belongs under `src/apps`.
