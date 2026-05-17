# fa_low_pass

`fa_low_pass` is a dedicated frequency processing node. It applies a first-order
low-pass filter to `FLOAT32LE` interleaved `AudioFrame` samples and publishes a
new stream.

It does not resample, convert bit depth, change channel count, normalize, apply
gain, limit, denoise, or open audio devices. Startup configuration errors fail
closed. Runtime frames that do not match the configured contract are dropped
with a warning.

```bash
ros2 launch fa_low_pass fa_low_pass.launch.py
```

See `docs/仕様書.md` for the runtime contract and `docs/テスト設計.md` for the
spec-to-test mapping.
