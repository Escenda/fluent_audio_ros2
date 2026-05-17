# src/io/utilities

I/O utility packages live here. These packages may record audio frames or send
them to network stream endpoints, but they are not device source/sink adapters
and do not own processing pipeline behavior.

`fa_stream` is a network sink utility. It is intentionally separate from
`src/streaming`, which is reserved for transport-stability nodes such as jitter
buffering, clock drift correction, PLC, and time alignment.
