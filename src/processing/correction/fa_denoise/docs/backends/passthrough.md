# passthrough backend

## 目的

`passthrough` は `fa_denoise` の topic wiring と diagnostics を確認するための backend である。denoise は行わない。

default config では選択しない。使用する場合は debug / wiring validation 用 config で `backend.name: "passthrough"` を明示する。

## 入力

- `stream_id` が `input_topic` と一致する validated `AudioFrame`

## 出力

- 入力と同じ `AudioFrame`

## 注意

model load failure や unknown backend を `passthrough` に変換してはならない。`enabled=false` も pass-through ではなく drop として扱う。
