# fa_vad

`fa_vad` runs local Silero VAD inference on a continuous 16 kHz mono
`AudioFrame` stream and publishes `VoiceActivity` probability/state frames.

The model backend only computes speech probability for fixed windows.
`VadProbabilityStream` handles PCM buffering and model-window scheduling.
`SpeechActivityEstimator` handles hysteresis and probability-delta based
speech start/end estimation. The ROS node only validates `AudioFrame` metadata,
composes those objects, and publishes `VoiceActivity`.
