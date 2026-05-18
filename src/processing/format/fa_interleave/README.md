# fa_interleave

`fa_interleave` は `fa_interfaces/msg/AudioFrame` の memory layout だけを変換する FluentAudio processing package です。

この package は IO node ではありません。device、file、network、codec decode/encode、resample、sample format conversion、bit-depth conversion、channel count conversion、gain、loudness normalize、limit、filter は扱いません。
layout reorder engine は ROS2 topic/message を知らない `internal_layout_reorder` backend に分離し、node は parameter、QoS、AudioFrame metadata、publish/drop、diagnostics だけを扱います。

## 対応変換

- `interleaved` -> `planar`
- `planar` -> `interleaved`

対応 sample format は次の明示的な組に限定します。

- `FLOAT32LE` / 32 bit
- `PCM16LE` / 16 bit
- `PCM32LE` / 32 bit

## 入出力契約

- subscribe: `input_topic`
- publish: `output_topic`
- ROS topic と `AudioFrame.stream_id` は別 identity として扱います。入力 stream は `input_stream_id`、出力 stream は `output.stream_id` で明示します。
- 入力 frame の `stream_id` は `input_stream_id` と一致する必要があります。
- 入力 metadata は `expected.sample_rate`、`expected.channels`、`expected.encoding`、`expected.bit_depth`、`input.layout` と一致する必要があります。
- `expected.channels` は `> 0` である必要があります。
- 出力では `source_id`、`header`、`sample_rate`、`channels`、`encoding`、`bit_depth`、`epoch` を保持します。
- 出力では `stream_id` を `output.stream_id` に更新し、`layout` を `output.layout` に更新します。

`input_stream_id` と `output.stream_id` は空文字、raw/resolved ROS topic と同一、相互に同一のいずれも禁止です。

契約に合わない runtime frame は publish せず warning を出して drop します。起動時に unsupported layout / format config や stream identity collision が指定された場合は fail closed で起動失敗します。
