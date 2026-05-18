# network_pcm_sender backend

`network_pcm_sender` は incoming `AudioFrame` payload を明示 network endpoint へ送信する backend contract である。

## Required Config

- `backend.name`
- `endpoint.uri`
- `transport.identity`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.bit_depth`
- `expected.layout`

## Forbidden

- hidden jitter buffer
- hidden packet loss concealment
- hidden clock drift correction
- hidden codec encode
- endpoint guessing

## Runtime Boundary

The backend is a ROS-free UDP sender. It accepts only explicit
`udp://<IPv4>:<port>` endpoints, opens one UDP socket, and sends each incoming
payload as one datagram.

It does not resolve DNS names, retry, reorder packets, add packet headers,
perform clock drift correction, or switch to an alternate endpoint.

## Failure Policy

- empty `endpoint.uri`: fail closed
- non-UDP endpoint: fail closed
- non-IPv4 host: fail closed
- invalid port: fail closed
- socket creation failure: fail closed
- send failure or partial send: fail closed
