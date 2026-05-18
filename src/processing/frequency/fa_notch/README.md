# fa_notch

`fa_notch` is a dedicated frequency processing node. It applies a second-order
notch biquad filter to `FLOAT32LE` `AudioFrame` samples and publishes a new
stream.

It does not resample, convert bit depth, change channel count, normalize, apply
gain, limit, or open audio devices. Frames that do not match the configured
`FLOAT32LE` / 32-bit / interleaved contract are dropped.

ROS topics are transport identities only. The accepted input
`AudioFrame.stream_id` is configured by `input_stream_id`, and the published
`AudioFrame.stream_id` is configured by `output.stream_id`.

```bash
ros2 launch fa_notch fa_notch.launch.py
```

See `docs/仕様書.md` for the runtime contract and `docs/テスト設計.md` for the
spec-to-test mapping.
