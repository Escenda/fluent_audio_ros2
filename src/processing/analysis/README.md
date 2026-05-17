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
