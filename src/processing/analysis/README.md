# Analysis Processing

This category contains nodes that turn audio into decisions, transcripts,
features, or model-input representations.

Examples:

- VAD
- KWS
- ASR
- turn detection
- STFT
- Mel spectrogram
- MFCC
- loudness measurement
- speaker embedding
- audio embedding

Inference engines live behind package-local backend boundaries. Backends do not
know ROS 2 topics or messages; nodes adapt ROS 2 contracts to backend contracts.

## Package Status

Only directories with `package.xml` are ROS 2 packages.

| Directory | Status |
| --- | --- |
| `fa_vad/` | ROS 2 package |
| `fa_kws/` | ROS 2 package |
| `fa_asr/` | ROS 2 package |
| `fa_turn_detector/` | ROS 2 package |
| `fa_audio_embedding/` | roadmap placeholder; not a ROS 2 package |
| `fa_log_mel/` | roadmap placeholder; not a ROS 2 package |
| `fa_sed/` | roadmap placeholder; not a ROS 2 package |
| `fa_speaker/` | roadmap placeholder; not a ROS 2 package |
