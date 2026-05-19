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

Network endpoint sinks such as `fa_stream` live under `src/io/utilities`; they
may publish audio to an external stream, but they are not transport-stability
nodes.

## Package Status

Only directories with `package.xml` are ROS 2 packages.

| Directory | Status |
| --- | --- |
| `fa_chunk_overlap/` | ROS 2 package |
| `fa_clock_drift/` | ROS 2 package |
| `fa_frame_buffer/` | ROS 2 package |
| `fa_jitter_buffer/` | ROS 2 package |
| `fa_latency_compensation/` | ROS 2 package |
| `fa_overlap_add/` | ROS 2 package |
| `fa_packet_loss_concealment/` | ROS 2 package |
| `fa_time_alignment/` | ROS 2 package |
