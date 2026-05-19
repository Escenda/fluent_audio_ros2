# file_writer backend

## 目的

`file_writer` は validated `AudioFrame` chunk を file へ保存する backend である。ROS2 topic、profile、system config は知らない。

## 入力

- output path
- `AudioFormat.encoding`
- `AudioFormat.sample_rate`
- `AudioFormat.channels`
- `AudioFormat.bit_depth`
- PCM chunk

## 対応 format

- `PCM16LE` / 16 bit
- `FLOAT32LE` / 32 bit

いずれも positive `sample_rate`、positive `channels`、`interleaved` layout の chunk を前提にする。recording session 中は `encoding` / `bit_depth` / `sample_rate` / `channels` / `layout` を固定し、file header と payload contract を同じ format で維持する。resample、downmix、channel conversion、bit-depth conversion、format conversion は行わない。

unsupported format、format change、empty chunk、file contract に合わない payload は explicit error result にする。失敗を隠すために silent conversion や別 format への書き換えはしない。

## 失敗条件

- output path が空
- parent directory が存在しない、または書けない
- unsupported encoding
- recording session 中の format change
- write / flush failure

失敗時に別 path へ書き込む fallback は禁止する。
