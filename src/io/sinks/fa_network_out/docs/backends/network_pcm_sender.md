# network_pcm_sender backend

`network_pcm_sender` は incoming `AudioFrame` payload を明示 network endpoint へ送信する backend contract である。

## Required Config

- `backend.name`
- `endpoint.uri`
- `transport.identity`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.layout`

## Forbidden

- hidden jitter buffer
- hidden packet loss concealment
- hidden clock drift correction
- hidden codec encode
- endpoint guessing
