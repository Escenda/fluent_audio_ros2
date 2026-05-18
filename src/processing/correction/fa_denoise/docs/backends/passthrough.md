# passthrough backend

## 目的

`passthrough` は `fa_denoise` の wiring と diagnostics を確認するための ROS-free identity backend である。denoise は行わない。

default config では選択しない。使用する場合は debug / wiring validation 用 config で `backend.name: "passthrough"` を明示する。

## 入力

- `AudioFormat`
- PCM bytes

## 出力

- 入力と同じ PCM bytes
- 入力と同じ `AudioFormat`

backend は ROS2 topic、`AudioFrame`、diagnostics を知らない。`stream_id` の検証と output `stream_id` の付け替えは node の責務である。

## 注意

model load failure や unknown backend を `passthrough` に変換してはならない。`enabled=false` も pass-through ではなく drop として扱う。
