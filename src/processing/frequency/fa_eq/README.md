# fa_eq

`fa_eq` は FluentAudio の `fa_interfaces/msg/AudioFrame` を購読し、FLOAT32LE interleaved 音声へ明示的な 3-band EQ を適用して publish する ROS2 processing package です。

## Contract

- Node: `fa_eq`
- Executable: `fa_eq_node`
- Input: `input_topic`
- Output: `output_topic`
- 入力契約: `source_id` 非空、`stream_id == input_topic`、`sample_rate > 0`、`channels > 0`、`encoding == FLOAT32LE`、`bit_depth == 32`、`layout == interleaved`
- データ契約: 非空、`channels * sizeof(float)` で割り切れる、各サンプルが finite normalized `[-1, 1]`

起動時設定が不正な場合は fail closed します。runtime frame が契約を満たさない場合、または EQ 後の出力が normalized `[-1, 1]` を外れる場合は warning を出して drop します。clamp、normalize、resampling、device I/O は行いません。

## Parameters

`config/default.yaml` の既定値:

- `input_topic`: `audio/sample_format/mic`
- `output_topic`: `audio/eq/mic`
- `low.cutoff_hz`: `250.0`
- `high.cutoff_hz`: `4000.0`
- `gains.low_db`: `0.0`
- `gains.mid_db`: `0.0`
- `gains.high_db`: `0.0`
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

`/diagnostics` に `low_cutoff_hz`、`high_cutoff_hz`、`low_alpha`、`high_alpha`、各 band gain、`state_source_id`、`source_resets`、`frames_in`、`frames_out`、`frames_dropped`、`output_topic` を publish します。
