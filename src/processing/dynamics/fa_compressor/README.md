# fa_compressor

`fa_compressor` は FluentAudio の `fa_interfaces/msg/AudioFrame` を入力し、FLOAT32LE interleaved stream に static per-sample dynamic range compression を適用する ROS2 package です。

## 責務

- `input_topic` から `AudioFrame` を subscribe する。
- `output_topic` へ圧縮済み `AudioFrame` を publish する。
- `source_id`、format metadata、`header`、`epoch` を維持し、`stream_id` のみ `output.stream_id` に更新する。
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

## 明示設定例

`config/default.yaml` は launch fallback ではなく、明示して渡す設定例である。

- input topic: `fa_compressor/input`
- output topic: `fa_compressor/output`
- input stream: `audio/normalized/mic`
- output stream: `audio/compressed/mic`
- threshold: `0.5`
- ratio: `4.0`
- makeup gain: `1.0`
- expected format: `16000Hz`, `1ch`, `FLOAT32LE`, `32bit`, `interleaved`

## 起動

```bash
ros2 launch fa_compressor fa_compressor.launch.py node_name:=fa_compressor config_file:=/path/to/fa_compressor.yaml
```
