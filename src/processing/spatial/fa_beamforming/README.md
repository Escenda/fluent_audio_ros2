# fa_beamforming

`fa_beamforming` は FluentAudio の `AudioFrame` を対象に、FLOAT32LE / interleaved の multi-channel 入力へ明示的な固定 weight を適用し、mono stream を publish する ROS2 spatial processing package です。

## 役割

- `beamforming.weights` を channel ごとの固定 delay-and-sum style weight として扱う
- 入力の `sample_rate`、`channels`、`encoding`、`bit_depth`、`layout` が `expected.*` と一致することを検証する
- `source_id`、`header`、`sample_rate`、`epoch` を保持する
- `stream_id` を `output.stream_id` に更新し、`channels=1`、`encoding=FLOAT32LE`、`bit_depth=32`、`layout=interleaved`、`data` を出力契約へ更新する
- config と frame counter を `diagnostics` に publish する

ROS topic は搬送路であり、`AudioFrame.stream_id` ではありません。入力 stream は
`input_stream_id`、出力 stream は `output.stream_id` で明示します。

## 非役割

この package は weight 推定、equal-weight fallback、resampling、sample format conversion、gain normalize、limiter、AEC、denoise、source separation、device I/O を行いません。

## 起動例

```bash
ros2 launch fa_beamforming fa_beamforming.launch.py
```
