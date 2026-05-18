# fa_network_out

`fa_network_out` is a ROS 2 sink adapter that sends incoming
`fa_interfaces/msg/AudioFrame` payload bytes to an explicitly configured UDP
endpoint.

It does not encode media, resample, change channel layout, change sample format,
add a jitter buffer, perform packet loss concealment, or correct clock drift.
Those steps must be explicit processing or `src/streaming` nodes.

## Launch

```bash
ros2 launch fa_network_out fa_network_out.launch.py \
  node_name:=fa_network_out \
  config_file:=/path/to/fa_network_out.yaml
```

The default config intentionally leaves `endpoint.uri` and `transport.identity`
empty, so direct default launch fails closed until a site or test config binds a
concrete endpoint.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/network_pcm_sender.md`
