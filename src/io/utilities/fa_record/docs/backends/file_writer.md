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

いずれも interleaved chunk を前提にする。resample、channel conversion、bit-depth conversion は行わない。

## 失敗条件

- output path が空
- parent directory が存在しない、または書けない
- unsupported encoding
- recording session 中の format change
- write / flush failure

失敗時に別 path へ書き込む fallback は禁止する。
