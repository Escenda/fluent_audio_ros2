# internal_feedback_echo backend

## 概要

`internal_feedback_echo` は外部 DSP engine を使わず、package 内部の ring delay line だけで
feedback echo を処理する ROS 非依存 backend である。ROS topic、`AudioFrame` message、
diagnostics は知らない。

## Contract

- 入力 sample は `FLOAT32LE` interleaved の正規化 float
- channel 数は `expected.channels`
- delay line は channel ごとに独立
- source 切替時に delay line を silence 初期化
- 失敗 frame では output buffer と delay state を commit しない
- clipping、limiting、gain normalization は行わない

## 計算式

```text
delayed = delay_line[channel][position]
output = dry_gain * input + wet_gain * delayed
next_delay_state = input + feedback_gain * delayed
delay_line[channel][position] = next_delay_state
position = (position + 1) % delay_samples
```

## 安全条件

`feedback_gain` は起動時に finite かつ `abs(value) < 1.0` として検証する。これは feedback loop が設定だけで無制限に増幅される状態を拒否するためであり、入力や wet/dry gain を暗黙に補正するものではない。

output または next delay state が `[-1.0, 1.0]` 外になった場合、値を丸めず `ProcessStatus`
で拒否する。drop と diagnostics counter 更新は ROS node 側の責務である。
