# fa_compressor

`fa_compressor` は FluentAudio の `fa_interfaces/msg/AudioFrame` を入力し、FLOAT32LE interleaved stream に static per-sample dynamic range compression を適用する ROS2 package です。

## 責務

- `input_topic` から `AudioFrame` を subscribe する。
- `output_topic` へ圧縮済み `AudioFrame` を publish する。
- `source_id`、format metadata、`header`、`epoch` を維持し、`stream_id` のみ `output_topic` に更新する。
- 起動時 config が不正な場合は fail closed する。
- runtime frame が契約に合わない場合は warning を出して drop する。

## 非責務

- device I/O
- resampling
- sample format conversion
- channel conversion
- limiter / gate / normalize / filter / denoise
- attack / release envelope
- clamp / limit

## 既定設定

- input: `audio/normalized/mic`
- output: `audio/compressed/mic`
- threshold: `0.5`
- ratio: `4.0`
- makeup gain: `1.0`
- expected format: `16000Hz`, `1ch`, `FLOAT32LE`, `32bit`, `interleaved`

## 起動

```bash
ros2 launch fa_compressor fa_compressor.launch.py
```
