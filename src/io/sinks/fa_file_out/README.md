# fa_file_out

`fa_file_out` is a ROS 2 sink adapter that writes incoming
`fa_interfaces/msg/AudioFrame` payloads to an explicitly configured raw PCM file.

It does not encode media containers, resample, change channel layout, change
sample format, normalize, limit, or adjust gain. Those steps must be explicit
processing nodes such as `fa_encode`, `fa_resample`, `fa_channel_convert`,
`fa_sample_format`, or `fa_limiter`.

## Launch

```bash
ros2 launch fa_file_out fa_file_out.launch.py \
  node_name:=fa_file_out \
  config_file:=/path/to/fa_file_out.yaml
```

The default config intentionally leaves `file.path` empty, so direct default
launch fails closed until a site or test config binds a concrete target.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/pcm_file_writer.md`
