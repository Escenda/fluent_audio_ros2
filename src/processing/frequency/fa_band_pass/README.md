# fa_band_pass

`fa_band_pass` は FluentAudio の `fa_interfaces/msg/AudioFrame` を購読し、FLOAT32LE interleaved 音声へ一次 high-pass 後に一次 low-pass を適用して publish する ROS2 processing package です。

## Contract

- Node: `fa_band_pass`
- Executable: `fa_band_pass_node`
- Input topic: `input_topic`
- Output topic: `output_topic`
- Input stream identity: `input_stream_id`
- Output stream identity: `output.stream_id`
- 入力契約: `source_id` 非空、`stream_id == input_stream_id`、`sample_rate > 0`、`channels > 0`、`encoding == FLOAT32LE`、`bit_depth == 32`、`layout == interleaved`
- データ契約: 非空、`channels * sizeof(float)` で割り切れる、各サンプルが finite normalized `[-1, 1]`

起動時設定が不正な場合は fail closed します。runtime frame が契約を満たさない場合は warning を出して drop し、意味を変える fallback、clamp、normalize、resampling、device I/O は行いません。
ROS topic は搬送路の identity であり、`AudioFrame.stream_id` とは分離します。

## Parameters

`config/default.yaml` の既定値:

- `input_topic`: `fa_band_pass/input`
- `output_topic`: `fa_band_pass/output`
- `input_stream_id`: `audio/sample_format/mic`
- `output.stream_id`: `audio/band_pass/mic`
- `filter.low_cut_hz`: `80.0`
- `filter.high_cut_hz`: `3400.0`
- `expected.sample_rate`: `16000`
- `expected.channels`: `1`
- `expected.encoding`: `FLOAT32LE`
- `expected.bit_depth`: `32`
- `expected.layout`: `interleaved`
- `qos.depth`: `10`
- `qos.reliable`: `false`
- `diagnostics.publish_period_ms`: `1000`
- `diagnostics.qos.depth`: `10`
- `diagnostics.qos.reliable`: `true`

## Diagnostics

`/diagnostics` に `input_topic`、`output_topic`、`input_stream_id`、`output_stream_id`、`filter_low_cut_hz`、`filter_high_cut_hz`、`hp_alpha`、`lp_alpha`、`state_source_id`、`state_resets`、`frames_in`、`frames_out`、`frames_dropped` を publish します。
