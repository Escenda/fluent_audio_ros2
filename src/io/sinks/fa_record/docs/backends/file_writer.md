# file_writer backend

## 目的

`file_writer` は validated `AudioFrame` chunk を file へ保存する backend である。ROS2 topic、profile、system config は知らない。

## 入力

- output path
- expected encoding
- expected sample rate
- expected channels
- expected bit depth
- PCM chunk

## 失敗条件

- output path が空
- parent directory が存在しない、または書けない
- unsupported encoding
- recording session 中の format change
- write / flush failure

失敗時に別 path へ書き込む fallback は禁止する。
