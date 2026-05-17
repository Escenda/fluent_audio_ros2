# fa_deesser

`fa_deesser` is a dedicated frequency processing node. It applies a first-order
split-band de-esser to `FLOAT32LE` interleaved `AudioFrame` samples and
publishes a new stream.

The node does not open audio devices, resample, convert sample format, change
channel count, normalize, compress, limit, denoise, or perform voice activity
detection. It splits each channel into low and high bands, detects excessive
high-band magnitude, attenuates only the high band, and recombines the bands.

```bash
ros2 launch fa_deesser fa_deesser.launch.py
```

See `docs/仕様書.md` for the runtime contract and `docs/テスト設計.md` for the
spec-to-test mapping.
