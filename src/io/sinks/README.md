# src/io/sinks

Output sink adapters live here. They subscribe to audio frames and write them to
explicit sinks without hidden format conversion, gain, limiting, or routing.
`fa_out` owns local speaker, raw PCM file, and raw PCM UDP sink backends. It does
not hide jitter buffering, packet loss concealment, clock drift correction,
encoding, gain, or routing inside those sink adapters.
