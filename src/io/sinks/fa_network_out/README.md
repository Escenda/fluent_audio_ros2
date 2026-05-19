# fa_network_out

`fa_network_out` is the design-map directory for a future standalone network
sink adapter.

This is not a ROS 2 package yet. Do not add `package.xml` until packetization,
endpoint, jitter/clock interaction, and launch contracts are specified and
covered by tests.

Current FluentAudio profiles must not enable `package: fa_network_out`. Raw PCM
UDP output is currently handled by `fa_out` through its explicit
`network_pcm_sender` sink backend.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/network_pcm_sender_adapter.md`
