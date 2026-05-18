# internal_layout_reorder

## 目的

`internal_layout_reorder` は validated audio byte sequence を `interleaved` と `planar` の間で byte block reorder する内部 backend である。
ROS2 topic/message を知らず、byte列と `FrameContract` を受け取り、byte列と `ProcessResult` を返す。

## 入力

- byte sequence
- `FrameContract`
- `bytes_per_sample`
- `channels`
- `input.layout`
- `output.layout`

## 出力

- reordered byte vector
- `ProcessResult`

## 境界

この backend は ROS2 topic、QoS、diagnostics、profile、device、codec を知らない。sample 値を数値として解釈せず、`bytes_per_sample` 単位でコピーする。

## Fail closed

起動時 config が未対応の場合は node 起動を失敗させる。runtime frame の data size が frame 境界を満たさない場合は `FrameContractStatus` で拒否し、caller が warning を出して drop する。padding、zero fill、truncate は行わない。
backend は失敗時に出力 buffer を更新しないため、呼び出し元は古い変換結果を誤って publish しない。
