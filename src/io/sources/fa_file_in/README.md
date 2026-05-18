# fa_file_in

`fa_file_in` is a ROS 2 source adapter that reads an explicitly configured raw
PCM file and publishes its bytes as `fa_interfaces/msg/AudioFrame`.

It does not decode encoded media, resample, change channel layout, change sample
format, or adjust gain. Those steps must be explicit processing nodes such as
`fa_decode`, `fa_resample`, `fa_channel_convert`, or `fa_sample_format`.

## Launch

```bash
ros2 launch fa_file_in fa_file_in.launch.py \
  node_name:=fa_file_in \
  config_file:=/path/to/fa_file_in.yaml
```

The default config intentionally leaves `file.path` empty, so direct default
launch fails closed until a site or test config binds a concrete fixture.

## Documents

- `docs/仕様書.md`
- `docs/アルゴリズム詳細説明書.md`
- `docs/テスト設計.md`
- `docs/backends/pcm_file_reader.md`
