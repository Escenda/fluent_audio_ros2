# no_runtime_backend

`fa_voice_command_router` does not select an audio/model runtime backend.

The node is an application-layer router. It consumes command text, updates
router state, optionally calls the configured TTS service, and optionally
publishes an output stop signal.

## Boundary

- No VAD/KWS/ASR/TD inference runs in this package.
- No TTS engine is imported or selected here.
- `tts_service` is a ROS service name, not a backend selector.
- Missing required topics/services are configuration errors at node startup.

## Failure Policy

If `announce_tts=true`, `tts_service` must be non-empty. If
`stop_output_on_stop=true`, `output_stop_topic` must be non-empty. The router
does not silently disable those behaviors to keep running.
