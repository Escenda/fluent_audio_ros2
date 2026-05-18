# fa_sample_format

`fa_sample_format` は `fa_interfaces/msg/AudioFrame` の sample representation だけを変換する FluentAudio processing package です。

この package は IO node ではありません。device、file、network、codec decode/encode、resample、gain、loudness normalize、limit、filter、channel count 変更は扱いません。
変換 engine は ROS2 topic/message を知らない `internal_float32le` backend に分離し、node は parameter、QoS、AudioFrame metadata、publish/drop、diagnostics だけを扱います。

## 初期対応変換

- `PCM16LE` / 16 bit / `interleaved` -> `FLOAT32LE` / 32 bit / `interleaved`
- `PCM32LE` / 32 bit / `interleaved` -> `FLOAT32LE` / 32 bit / `interleaved`
- `FLOAT32LE` / 32 bit / `interleaved` -> `PCM16LE` / 16 bit / `interleaved`

変換は config で明示された組み合わせだけを受け付けます。frame metadata から自動判定しません。`FLOAT32LE -> PCM16LE` では non-finite または `[-1.0, 1.0]` 範囲外 sample を clamp せず drop します。

## 入出力契約

- subscribe: `input_topic`
- publish: `output_topic`
- 入力 metadata は `input.encoding`、`input.bit_depth`、`expected.sample_rate`、`expected.channels`、`expected.layout` と一致する必要があります。
- `expected.layout` は `interleaved` のみ対応します。
- 出力では `source_id`、`sample_rate`、`channels`、`layout`、`header`、`epoch` を保持します。
- 出力では `stream_id` を `output_topic` に更新し、`encoding` / `bit_depth` を出力形式へ更新します。

契約に合わない runtime frame は publish せず drop します。起動時に unsupported conversion が指定された場合は fail closed で起動失敗します。
