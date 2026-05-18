# src/io/sinks

Output sink adapters live here. They subscribe to audio frames and write them to
explicit sinks without hidden format conversion, gain, limiting, or routing.
`fa_out` owns local speaker and raw PCM file sink backends. Network sinks remain
separate until their transport contract is folded into the same backend boundary.
