# fa_network_in

`fa_network_in` is the design-map directory for a future standalone network
source adapter.

This is not a ROS 2 package yet. Do not add `package.xml` until jitter,
packet-loss, clock-drift, endpoint, and launch contracts are specified and
covered by tests.

Current FluentAudio profiles must not enable `package: fa_network_in`. Raw PCM
UDP input is currently handled by `fa_in` through its explicit
`network_pcm_receiver` source backend.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/network_pcm_receiver_adapter.md`
