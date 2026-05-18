# network_pcm_receiver backend

`network_pcm_receiver` は明示 network endpoint から PCM packet を受け取る backend contract である。

## Required Config

- `backend.name`
- `endpoint.uri`
- `expected.sample_rate`
- `expected.channels`
- `expected.encoding`
- `expected.layout`

## Forbidden

- hidden jitter buffer
- hidden packet loss concealment
- hidden codec decode
- endpoint guessing

