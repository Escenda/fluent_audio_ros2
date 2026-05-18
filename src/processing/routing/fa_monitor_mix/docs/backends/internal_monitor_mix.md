# internal_monitor_mix backend

## 目的

`internal_monitor_mix` は ROS-free な monitor bus mix backend である。ROS2 topic、`AudioFrame`、diagnostics は知らず、入力 PCM bytes と gain config だけを扱う。

## 入力

- input count
- master index
- channel count
- input gains
- 各入力の `FLOAT32LE` interleaved PCM bytes

## 出力

- mix 後の `FLOAT32LE` interleaved PCM bytes
- explicit `ProcessStatus`

## fail-closed 条件

- 入力数不一致
- empty / misaligned input
- byte length mismatch
- input sample が finite normalized range を満たさない
- mix output sample が finite normalized range を満たさない

backend は silence 補完、clamp、limiter、normalize、resample を行わない。
