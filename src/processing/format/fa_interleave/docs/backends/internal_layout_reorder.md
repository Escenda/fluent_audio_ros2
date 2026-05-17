# internal_layout_reorder

## 目的

`internal_layout_reorder` は validated `AudioFrame.data` を `interleaved` と `planar` の間で byte block reorder する内部 backend である。

## 入力

- validated `AudioFrame`
- `bytes_per_sample`
- `channels`
- `input.layout`
- `output.layout`

## 出力

- reordered byte vector

## 境界

この backend は ROS2 topic、QoS、diagnostics、profile、device、codec を知らない。sample 値を数値として解釈せず、`bytes_per_sample` 単位でコピーする。

## Fail closed

起動時 config が未対応の場合は node 起動を失敗させる。runtime frame の data size が frame 境界を満たさない場合は caller が warning を出して drop する。padding、zero fill、truncate は行わない。
