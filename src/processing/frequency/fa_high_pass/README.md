# fa_high_pass

`fa_high_pass` is a dedicated frequency processing node. It applies a first-order
high-pass filter to `FLOAT32LE` `AudioFrame` samples and publishes a new stream.

It does not resample, convert bit depth, change channel count, normalize, apply
gain, limit, or open audio devices. Frames that do not match the configured
`FLOAT32LE` / 32-bit / interleaved contract are dropped.
ROS topic names are transport identities only. Input frames must carry
`AudioFrame.stream_id == input_stream_id`, and output frames use
`output.stream_id`.

```bash
ros2 launch fa_high_pass fa_high_pass.launch.py
```

See `docs/仕様書.md` for the runtime contract and `docs/テスト設計.md` for the
spec-to-test mapping.
