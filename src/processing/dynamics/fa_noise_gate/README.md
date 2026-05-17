# fa_noise_gate

`fa_noise_gate` is a dedicated dynamics processing node. It applies an explicit
threshold noise gate to `FLOAT32LE` interleaved `AudioFrame` samples and
publishes a new stream.

It does not open audio devices, resample, convert sample format, change channel
count, compress, limit, normalize, filter, or denoise. Samples with absolute
amplitude below `gate.threshold_linear` are multiplied by
`gate.closed_gain_linear`; all other valid samples are preserved.

```bash
ros2 launch fa_noise_gate fa_noise_gate.launch.py
```

See `docs/仕様書.md` for the runtime contract and `docs/テスト設計.md` for the
spec-to-test mapping.
