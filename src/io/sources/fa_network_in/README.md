# fa_network_in

`fa_network_in` is the network source adapter that receives raw PCM audio from
an explicit UDP endpoint and publishes its contract as `AudioFrame`.

This package is source-only. Jitter buffering, clock drift correction, packet
loss concealment, codec decode, resampling, gain, and format negotiation belong
in explicit downstream processing or streaming nodes, not inside this adapter.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/network_pcm_receiver.md`

## Runtime

- `launch/fa_network_in.launch.py`
- `config/default.yaml`
- executable: `fa_network_in_node`
