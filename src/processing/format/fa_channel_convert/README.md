# fa_channel_convert

`fa_channel_convert` は、FluentAudio の `AudioFrame` を対象に、FLOAT32LE / interleaved の channel count だけを変換する ROS2 processing package です。

## 役割

- `mono_to_stereo_duplicate`: 1ch 入力を 2ch 出力へ複製する
- `stereo_to_mono_average`: 2ch 入力を 1ch 出力へ平均する
- `source_id`、`header`、`sample_rate`、`bit_depth`、`encoding`、`layout`、`epoch` を保持する
- `stream_id` を `output.stream_id` に更新し、`channels` と `data` を変換結果に更新する
- config と frame counter を `diagnostics` に publish する

ROS topic は搬送路であり、`AudioFrame.stream_id` ではありません。入力 stream は
`input_stream_id`、出力 stream は `output.stream_id` で明示します。

## 非役割

この package は device I/O、resampling、sample format conversion、gain、limiter、noise gate、filtering、denoise を行いません。

## 起動例

```bash
ros2 launch fa_channel_convert fa_channel_convert.launch.py
```
