# network_pcm_receiver backend

`network_pcm_receiver` は明示 network endpoint から PCM packet を受け取る backend contract である。

## Required Config

- `backend.name`
- `endpoint.uri`
- `transport.identity`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.bit_depth`
- `expected.layout`
- `network.max_packet_bytes`

## Runtime Boundary

- `endpoint.uri` は `udp://<IPv4>:<port>` のみ受け付ける。
- backend は UDP socket の open / bind / non-blocking receive だけを担当する。
- backend は ROS2 node、topic、message 型を知らない。
- datagram payload は byte 列として返し、decode、resample、channel conversion は行わない。

## Forbidden

- hidden jitter buffer
- hidden packet loss concealment
- hidden codec decode
- endpoint guessing
- DNS / hostname resolution
- retry / reorder / clock drift correction
