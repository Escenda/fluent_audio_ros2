# internal_float32le backend

`fa_sample_format` の初期 backend は PCM integer と `FLOAT32LE` の明示変換である。
ROS2 topic/message を知らず、byte列と `FrameContract` を受け取り、byte列と `ProcessResult` を返す。

## 責務

- `PCM16LE` / 16 bit sample を `FLOAT32LE` / 32 bit sample に変換する
- `PCM32LE` / 32 bit sample を `FLOAT32LE` / 32 bit sample に変換する
- `FLOAT32LE` / 32 bit sample を `PCM16LE` / 16 bit sample に変換する
- little-endian byte列を明示的に読む

## 非責務

- device I/O
- codec decode/encode
- resample
- channel count 変更
- gain / limiter / loudness normalize
- hidden clipping / clamp
- filtering
- non-interleaved layout 対応

## 失敗条件

入力 byte列が sample 幅で割り切れない場合は `FrameContractStatus` で拒否し、呼び出し元が frame を drop する。`FLOAT32LE -> PCM16LE` では non-finite または `[-1.0, 1.0]` 範囲外 sample も drop 対象であり、clamp しない。
backend は失敗時に出力 buffer を更新しないため、呼び出し元は古い変換結果を誤って publish しない。
