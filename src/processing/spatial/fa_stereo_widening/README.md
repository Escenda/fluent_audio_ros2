# fa_stereo_widening

`fa_stereo_widening` は FluentAudio の stereo spatial processing node です。`fa_interfaces/msg/AudioFrame` の FLOAT32LE interleaved stereo stream を購読し、mid/side transform による stereo width を適用して別 topic へ publish します。

## Contract

- package: `fa_stereo_widening`
- executable: `fa_stereo_widening_node`
- node: `fa_stereo_widening`
- input: `input_topic`
- output: `output_topic`
- expected frame: `sample_rate > 0`, `channels == 2`, `encoding == FLOAT32LE`, `bit_depth == 32`, `layout == interleaved`
- `source_id` と `stream_id` は必須
- 入力 `stream_id` は `input_topic` と一致する必要がある
- 出力 `stream_id` は `output_topic` に更新する

## Width

`width` は finite な `[0.0, 4.0]` の範囲で指定します。

- `0.0`: side を消して mono-compatible center にする
- `1.0`: 入力の stereo width を保持する
- `> 1.0`: side を増幅して stereo width を広げる

入力および出力 sample は finite な normalized `[-1, 1]` でなければならず、違反した frame は publish せず破棄します。clamp、normalize、default 補完は行いません。

## Non-goals

`fa_stereo_widening` は mid/side stereo width のみを担当します。device I/O、resampling、sample format conversion、channel-count conversion、limiter、filter、denoise、pan は行いません。
