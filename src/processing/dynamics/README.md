# Dynamics Processing

This category contains nodes that control amplitude, loudness, and dynamic
range.

Examples:

- gain
- normalize
- compressor
- limiter
- expander
- gate/noise gate
- automatic gain control

Dynamics processing is separate from device I/O. Microphone or speaker adapters
must not hide gain or AGC inside `fa_in` or `fa_out`.
