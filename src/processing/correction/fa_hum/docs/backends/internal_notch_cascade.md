# internal_notch_cascade backend

## 1. 役割

`internal_notch_cascade` は `fa_hum` node 内部で動作する C++ backend である。外部 process や model runtime は使わず、ROS2 topic/message を知らない。入力された `FLOAT32LE` sample bytes に対して deterministic な biquad notch cascade を適用する。

## 2. 入力

- `FLOAT32LE`
- interleaved layout
- finite normalized samples in `[-1.0, 1.0]`
- configured channel count に整列した byte列
- `source_id`
- `epoch`

`FLOAT32LE` の byte 解釈は little-endian target に限定する。non little-endian target は compile-time error とし、byteswap fallback は持たない。

## 3. 出力

- 入力と同じ sample 数の hum removal 後 `FLOAT32LE` bytes
- sample range は `[-1.0, 1.0]`

metadata copy、`stream_id` 更新、publish 抑止、diagnostics は ROS node 側の責務である。

## 4. state

state は channel ごと、notch stage ごとに以下を保持する。

- previous input 1
- previous input 2
- previous output 1
- previous output 2

`source_id` が変わると全 state をリセットする。同一 `source_id` で `epoch` が進んだ場合も全 state をリセットする。同一 `source_id` で `epoch` が前回より古い場合は stale epoch として拒否する。

## 5. fail closed policy

この backend は次の条件で処理を中止し、frame を publish しない。

- frame contract 不一致
- `source_id` が空
- byte length が channel frame に整列しない
- stale epoch
- 入力 sample が non-finite
- 入力 sample が normalized range 外
- stage 出力が non-finite
- 最終出力が normalized range 外
- `float` 変換後の出力が non-finite または normalized range 外

出力は clamp しない。range 外 output は異常な processing result として drop する。
