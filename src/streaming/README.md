# Streaming

`src/streaming` contains nodes that keep real-time audio transport stable. It is
top-level because buffering, clock, jitter, and packet-loss behavior are
transport contracts rather than DSP content transforms.

Examples:

- buffering
- jitter buffer
- clock drift correction
- packet loss concealment
- latency compensation
- time alignment
- chunk overlap
- overlap-add

Streaming nodes handle timing and transport behavior. They do not decide model
backend selection or device binding.
