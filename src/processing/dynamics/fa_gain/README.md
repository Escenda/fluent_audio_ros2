# fa_gain

`fa_gain` is a dedicated dynamics processing node. It applies explicit linear
gain to normalized `FLOAT32LE` `AudioFrame` samples and publishes a new stream.

It does not resample, convert bit depth, normalize, limit, compress, gate, or
open audio devices. Those responsibilities belong to separate processing or I/O
nodes.

```bash
ros2 launch fa_gain fa_gain.launch.py
```

See `docs/仕様書.md` for the runtime contract and `docs/テスト設計.md` for the
spec-to-test mapping.
