# fa_gain

`fa_gain` is a dedicated dynamics processing node. It applies explicit linear
gain to normalized `FLOAT32LE` `AudioFrame` samples and publishes a new stream.
ROS topic names and `AudioFrame.stream_id` are configured separately; topic
wiring is transport, while stream identity is part of the audio contract.

It does not resample, convert bit depth, normalize, limit, compress, gate, or
open audio devices. Those responsibilities belong to separate processing or I/O
nodes.

```bash
ros2 launch fa_gain fa_gain.launch.py node_name:=fa_gain config_file:=/path/to/fa_gain.yaml
```

See `docs/仕様書.md` for the runtime contract and `docs/テスト設計.md` for the
spec-to-test mapping.
