# fa_band_pass

`fa_band_pass` は FluentAudio の `fa_interfaces/msg/AudioFrame` を購読し、FLOAT32LE interleaved 音声へ一次 high-pass 後に一次 low-pass を適用して publish する ROS2 processing package です。

## Contract

- Node: `fa_band_pass`
- Executable: `fa_band_pass_node`
- Input: `input_topic`
- Output: `output_topic`
- 入力契約: `source_id` 非空、`stream_id == input_topic`、`sample_rate > 0`、`channels > 0`、`encoding == FLOAT32LE`、`bit_depth == 32`、`layout == interleaved`
- データ契約: 非空、`channels * sizeof(float)` で割り切れる、各サンプルが finite normalized `[-1, 1]`

起動時設定が不正な場合は fail closed します。runtime frame が契約を満たさない場合は warning を出して drop し、意味を変える fallback、clamp、normalize、resampling、device I/O は行いません。

## Parameters

`config/default.yaml` の既定値:

- `input_topic`: `audio/sample_format/mic`
- `output_topic`: `audio/band_pass/mic`
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

## Diagnostics

`/diagnostics` に `filter_low_cut_hz`、`filter_high_cut_hz`、`hp_alpha`、`lp_alpha`、`state_source_id`、`source_resets`、`frames_in`、`frames_out`、`frames_dropped`、`output_topic` を publish します。
