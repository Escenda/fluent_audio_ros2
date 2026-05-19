# network_pcm_sender_adapter

`network_pcm_sender_adapter` is the future backend contract for a standalone
`fa_network_out` package.

The current implementation lives in `fa_out` as the `network_pcm_sender` sink
backend. This document keeps the design-map directory explicit without declaring
a second ROS 2 package.

Backend responsibilities:

- connect or send to an explicit network endpoint
- validate raw PCM packet metadata
- send received chunks as raw payloads

Backend non-responsibilities:

- jitter buffering
- packet loss concealment
- clock drift correction
- codec encode
- resample / limiter / normalize
- ROS 2 topic or message handling
