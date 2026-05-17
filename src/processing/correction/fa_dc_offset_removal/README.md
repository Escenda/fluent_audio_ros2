# fa_dc_offset_removal

`fa_dc_offset_removal` は `fa_interfaces/msg/AudioFrame` の `FLOAT32LE` interleaved stream から、frame 単位の DC offset を除去する processing node です。

## I/O

| 種別 | 名前 | 型 |
| --- | --- | --- |
| subscribe | `input_topic` | `fa_interfaces/msg/AudioFrame` |
| publish | `output_topic` | `fa_interfaces/msg/AudioFrame` |
| publish | `diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` |

## 契約

- `sample_rate > 0`
- `channels > 0`
- `encoding == FLOAT32LE`
- `bit_depth == 32`
- `layout == interleaved`
- `source_id` と `stream_id` は空でない
- 入力 `stream_id` は `input_topic` と一致する
- 出力 `stream_id` は `output_topic` に更新する
- `data` は空でなく、`channels * sizeof(float)` で割り切れる

## 処理

各 frame でチャンネルごとの平均値を計算し、そのチャンネルの全 sample から差し引きます。clamp、normalize、gain、limiter、filter、resampling、sample format conversion、device I/O は行いません。

入力 sample、平均値、出力 sample のいずれかが非有限値になった場合、その frame は publish せず drop します。
