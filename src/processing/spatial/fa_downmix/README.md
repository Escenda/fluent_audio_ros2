# fa_downmix

`fa_downmix` は、FluentAudio の `AudioFrame` を対象に、FLOAT32LE / interleaved の明示的な N-channel downmix だけを行う ROS2 processing package です。

## 役割

- `average_to_mono`: 2ch 以上の入力を 1ch 出力へ全 channel 平均する
- `pair_average_to_stereo`: 偶数 4ch 以上の入力を 2ch 出力へ L/R pair 平均する
- `source_id`、`header`、`sample_rate`、`bit_depth`、`encoding`、`layout`、`epoch` を保持する
- `stream_id` を `output.stream_id` に更新し、`channels` と `data` を変換結果に更新する
- config と frame counter を `diagnostics` に publish する

ROS topic は搬送路であり、`AudioFrame.stream_id` ではありません。入力 stream は
`input_stream_id`、出力 stream は `output.stream_id` で明示します。

## 非役割

この package は upmix、device I/O、resampling、sample format conversion、gain、limiter、pan、filtering、denoise を行いません。

## 起動例

```bash
ros2 launch fa_downmix fa_downmix.launch.py
```
