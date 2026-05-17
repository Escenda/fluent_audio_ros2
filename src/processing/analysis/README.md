# Analysis Processing

This category contains non-AI feature extraction and measurement nodes.
Decision-making model nodes live in `src/ai`; transport stabilization lives in
`src/streaming`.

Examples:

- STFT
- Mel spectrogram
- MFCC
- loudness measurement

Feature extraction nodes may expose package-local backend boundaries when they
wrap native DSP libraries. Backends do not know ROS 2 topics or messages; nodes
adapt ROS 2 contracts to backend contracts.

## Package Status

Only directories with `package.xml` are ROS 2 packages.

| Directory | Status |
| --- | --- |
| `fa_log_mel/` | roadmap placeholder; not a ROS 2 package |
