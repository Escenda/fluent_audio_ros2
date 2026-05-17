# passthrough backend

## 目的

`passthrough` は `fa_denoise` の topic wiring と diagnostics を確認するための backend である。denoise は行わない。

## 入力

- validated `AudioFrame`

## 出力

- 入力と同じ `AudioFrame`

## 注意

model load failure や unknown backend を `passthrough` に変換してはならない。
