# fa_bit_depth

`fa_bit_depth` は `fa_interfaces/msg/AudioFrame` の PCM integer bit depth だけを変換する FluentAudio processing package です。

この package は IO node ではありません。device、file、network、codec decode/encode、sample format normalization、resample、gain、loudness normalize、limit、filter、channel count 変更は扱いません。
変換 engine は ROS2 topic/message を知らない `internal_integer_bit_depth` backend に分離し、node は parameter、QoS、AudioFrame metadata、publish/drop、diagnostics だけを扱います。

## 初期対応変換

- `PCM16LE` / 16 bit / `interleaved` -> `PCM32LE` / 32 bit / `interleaved`

変換は config で明示された組み合わせだけを受け付けます。frame metadata から自動判定しません。下位 bit を破棄する `PCM32LE/32 -> PCM16LE/16` は fail closed します。

## 入出力契約

- subscribe: `input_topic`
- publish: `output_topic`
- ROS topic と `AudioFrame.stream_id` は別 identity として扱います。入力 stream は `input_stream_id`、出力 stream は `output.stream_id` で明示します。
- 入力 metadata は `input.encoding`、`input.bit_depth`、`expected.sample_rate`、`expected.channels`、`expected.layout` と一致する必要があります。
- 入力 frame の `stream_id` は `input_stream_id` と一致する必要があります。
- `expected.layout` は `interleaved` のみ対応します。
- 出力では `source_id`、`sample_rate`、`channels`、`layout`、`header`、`epoch` を保持します。
- 出力では `stream_id` を `output.stream_id` に更新し、`encoding` / `bit_depth` を出力形式へ更新します。

`input_stream_id` と `output.stream_id` は空文字、ROS topic と同一、相互に同一のいずれも禁止です。
契約に合わない runtime frame は warning を出して publish せず drop します。起動時に unsupported conversion または不正な stream identity が指定された場合は fail closed で起動失敗します。
