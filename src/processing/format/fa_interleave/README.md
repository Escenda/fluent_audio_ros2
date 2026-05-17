# fa_interleave

`fa_interleave` は `fa_interfaces/msg/AudioFrame` の memory layout だけを変換する FluentAudio processing package です。

この package は IO node ではありません。device、file、network、codec decode/encode、resample、sample format conversion、bit-depth conversion、channel count conversion、gain、loudness normalize、limit、filter は扱いません。

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
- 入力 metadata は `expected.sample_rate`、`expected.channels`、`expected.encoding`、`expected.bit_depth`、`input.layout` と一致する必要があります。
- `expected.channels` は `> 0` である必要があります。
- 出力では `source_id`、`header`、`sample_rate`、`channels`、`encoding`、`bit_depth`、`epoch` を保持します。
- 出力では `stream_id` を `output_topic` に更新し、`layout` を `output.layout` に更新します。

契約に合わない runtime frame は publish せず warning を出して drop します。起動時に unsupported layout / format config が指定された場合は fail closed で起動失敗します。
