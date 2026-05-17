# fa_pan

`fa_pan` は FluentAudio の stereo spatial processing node です。`fa_interfaces/msg/AudioFrame` の FLOAT32LE interleaved stereo stream を購読し、constant-power pan を適用して別 topic へ publish します。

## Contract

- package: `fa_pan`
- executable: `fa_pan_node`
- node: `fa_pan`
- input: `input_topic`
- output: `output_topic`
- expected frame: `sample_rate > 0`, `channels == 2`, `encoding == FLOAT32LE`, `bit_depth == 32`, `layout == interleaved`
- `source_id` と `stream_id` は必須
- 入力 `stream_id` は `input_topic` と一致する必要がある
- 出力 `stream_id` は `output_topic` に更新する

## Pan

`pan.position` は `[-1.0, 1.0]` の範囲で指定します。

- `-1.0`: full left
- `0.0`: center
- `1.0`: full right

ゲインは `angle = (position + 1) * pi / 4` から `left_gain = cos(angle)`, `right_gain = sin(angle)` として計算します。入力および出力 sample は finite な normalized `[-1, 1]` でなければならず、違反した frame は publish せず破棄します。

## Non-goals

`fa_pan` は spatial placement のみを担当します。device I/O、resampling、sample format conversion、channel-count conversion、limiter、filter、denoise は行いません。
