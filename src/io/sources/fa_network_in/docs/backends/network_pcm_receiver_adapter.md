# network_pcm_receiver_adapter

`network_pcm_receiver_adapter` is the future backend contract for a standalone
`fa_network_in` package.

The current implementation lives in `fa_in` as the `network_pcm_receiver` source
backend. This document keeps the design-map directory explicit without declaring
a second ROS 2 package.

Backend responsibilities:

- bind an explicit network endpoint
- validate raw PCM packet metadata
- expose received chunks to the source node

Backend non-responsibilities:

- jitter buffering
- packet loss concealment
- clock drift correction
- codec decode
- resample / gain / denoise
- ROS 2 topic or message handling
