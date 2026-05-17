# fa_limiter

`fa_limiter` is a dedicated dynamics processing node. It applies an explicit
hard limiter to `FLOAT32LE` `AudioFrame` samples and publishes a new stream.

It does not resample, convert bit depth, change channel count, normalize, apply
gain, compress, gate, or open audio devices. Clamping samples to
`[-threshold.linear, +threshold.linear]` is the limiter's explicit
responsibility.

```bash
ros2 launch fa_limiter fa_limiter.launch.py
```

See `docs/仕様書.md` for the runtime contract and `docs/テスト設計.md` for the
spec-to-test mapping.
