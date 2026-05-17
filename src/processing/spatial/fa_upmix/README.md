# fa_upmix

`fa_upmix` は FluentAudio の `fa_interfaces/msg/AudioFrame` を入力し、FLOAT32LE interleaved のチャンネル数だけを明示的に増やす ROS2 processing node です。

## 対象範囲

- 入力: `FLOAT32LE`, `bit_depth=32`, `layout=interleaved`
- mode:
  - `mono_duplicate`: 1ch 入力を Nch 出力へ複製する。`N > 1`
  - `stereo_duplicate_pairs`: 2ch 入力を LR ペア単位で偶数 Nch 出力へ複製する。`N > 2`
- 出力: `stream_id` を `output_topic` に更新し、`channels` と `data` を upmix 後の値へ更新する

## 範囲外

- device I/O
- resample
- sample format conversion
- gain / limiter / normalize
- pan / spatial rendering
- channel label 推定

未対応 config は起動時に fail closed します。runtime frame が契約に合わない場合や sample が normalized finite range `[-1.0, 1.0]` を外れる場合は publish せず drop します。
