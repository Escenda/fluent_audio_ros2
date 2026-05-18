# no_runtime_backend

## backend scope

`fa_bus_router` は外部 runtime backend を持たない。処理は ROS2 subscription callback 内で完結し、入力 `AudioFrame` を output topic ごとに copy して publish する。

## 入力

- `fa_interfaces/msg/AudioFrame`
- `stream_id == input_stream_id`
- `sample_rate`、`channels`、`encoding`、`bit_depth`、`layout` が `expected.*` と一致
- `data.size()` は `channels * bit_depth / 8` の整数倍

## 出力

- output topic ごとに 1 copy
- `stream_id == output.stream_ids[index]`
- `data` は入力と同一 byte sequence
- `stream_id` 以外の metadata は入力と同一

## 禁止事項

- mixing
- resampling
- gain
- sample format conversion
- channel conversion
- filtering
- denoise
- invalid frame の補完
